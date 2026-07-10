import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

# ==========================================
# SHARED PARAMETERS
# ==========================================
START_CAPITAL = 100000
RISK_PER_TRADE = 0.01
MAX_SECTOR_POSITIONS = 2
MAX_TOTAL_POSITIONS = 6
MAX_ALLOC_FRACTION = 1/6

SLIPPAGE_ENTRY = 0.0005
SLIPPAGE_EXIT  = 0.0005
COMMISSION = 1.0
RISK_FREE_ANNUAL = 0.045
DAILY_RF = RISK_FREE_ANNUAL / 252

# Change these two to retest on a different market regime, e.g. the GFC
# (2007-01-01 to 2010-01-01) or a choppy period (2015-01-01 to 2016-06-01),
# without touching any other code.
START_DATE = "2015-01-01"
END_DATE = "2026-06-01"

tickers = [
    'MSFT', 'AMZN', 'JPM', 'UNH', 'XOM', 'WMT', 'NVDA', 'AMD', 'META', 'TSLA',
    'AAPL', 'GOOGL', 'NFLX', 'AVGO', 'CRM',  # Tech / Communications
    'HD', 'COST', 'NKE', 'MCD', 'ORLY',      # Consumer Discretionary / Staples
    'BAC', 'GS', 'MS', 'V', 'MA',            # Financials / Payments
    'LLY', 'MRK', 'PFE', 'JNJ',              # Healthcare
    'CVX', 'COP'                             # Energy
]

sectors = {
    'MSFT': 'Tech', 'NVDA': 'Tech', 'AMD': 'Tech', 'META': 'Tech',
    'AAPL': 'Tech', 'GOOGL': 'Tech', 'NFLX': 'Tech', 'AVGO': 'Tech', 'CRM': 'Tech',
    'AMZN': 'Consumer', 'TSLA': 'Consumer', 'WMT': 'Consumer',
    'HD': 'Consumer', 'COST': 'Consumer', 'NKE': 'Consumer', 'MCD': 'Consumer', 'ORLY': 'Consumer',
    'JPM': 'Finance', 'BAC': 'Finance', 'GS': 'Finance', 'MS': 'Finance',
    'V': 'Finance', 'MA': 'Finance',
    'UNH': 'Healthcare', 'LLY': 'Healthcare', 'MRK': 'Healthcare', 'PFE': 'Healthcare', 'JNJ': 'Healthcare',
    'XOM': 'Energy', 'CVX': 'Energy', 'COP': 'Energy'
}

# ==========================================
# LOAD DATA
# ==========================================
def load_data(start=START_DATE, end=END_DATE):
    print(f"Downloading historical data ({start} to {end})...")
    # FIXED: was auto_adjust=True, which reverted the earlier fix pinning
    # this to raw (unadjusted) close prices. auto_adjust=False keeps the
    # backtest on the SAME price basis as the live/forward-test script --
    # otherwise NVDA/TSLA/AMZN split-adjusted backtest prices would silently
    # diverge from the raw prices the live script actually trades on.
    data = yf.download(tickers + ['^GSPC'], start=start, end=end, auto_adjust=False)
    return data

# ==========================================
# BUILD INDICATORS
# ==========================================
def build_indicators(data):
    close = data['Close']
    high = data['High']
    low = data['Low']
    vol = data['Volume']

    ind = {}

    for t in tickers:
        if t not in close.columns:
            print(f"WARNING: '{t}' returned no data (delisted or not yet public in this window) -- skipping.")
            continue
        s = close[t]
        if s.isna().mean() > 0.05:
            print(f"WARNING: '{t}' has >5% missing history in this window -- skipping.")
            continue

        df = pd.DataFrame(index=close.index)
        df['Close'] = s
        df['High'] = high[t]
        df['Low'] = low[t]
        df['Vol'] = vol[t]

        df['SMA50'] = df['Close'].rolling(50).mean()
        df['SMA20'] = df['Close'].rolling(20).mean()
        df['EMA20'] = df['Close'].ewm(span=20).mean()
        df['SMA5'] = df['Close'].rolling(5).mean()
        df['SMA200'] = df['Close'].rolling(200).mean()

        df['AvgVol30'] = df['Vol'].rolling(30).mean()
        df['Prior20High'] = df['Close'].rolling(20).max().shift(1)

        tr1 = df['High'] - df['Low']
        tr2 = (df['High'] - df['Close'].shift(1)).abs()
        tr3 = (df['Low'] - df['Close'].shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df['ATR14'] = tr.rolling(14).mean()
        df['ATR50'] = tr.rolling(50).mean()

        delta = df['Close'].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(2).mean()
        avg_loss = loss.rolling(2).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        df['RSI2'] = 100 - (100 / (1 + rs))

        df['KC_MID'] = df['EMA20']
        df['KC_UPPER'] = df['EMA20'] + 1.5 * df['ATR14']
        df['KC_LOWER'] = df['EMA20'] - 1.5 * df['ATR14']

        ind[t] = df

    sp500 = close['^GSPC']
    sp500_sma200 = sp500.rolling(200).mean()
    sp500_sma50 = sp500.rolling(50).mean()

    return ind, sp500, sp500_sma200, sp500_sma50

# ==========================================
# SHARED UTILITIES
# ==========================================
def apply_entry_friction(price):
    return price * (1 + SLIPPAGE_ENTRY)

def apply_exit_friction(price):
    return price * (1 - SLIPPAGE_EXIT)

def apply_cash_drag(cash):
    return cash * (1 + DAILY_RF)

def gap_down_exit(open_price, stop_price, close_price):
    """If the day's open already gapped below the stop, fill at that
    (worse) gapped-down open. Otherwise hold through to that day's close
    (matches the live script's MOC exit routing)."""
    if open_price < stop_price:
        return open_price
    return close_price

def position_size(entry_price, stop_price, equity):
    friction_entry = apply_entry_friction(entry_price)
    risk_dist = friction_entry - stop_price
    if risk_dist <= 0:
        return 0
    risk_dollars = (equity * RISK_PER_TRADE) - COMMISSION
    shares_risk = risk_dollars / risk_dist
    max_alloc = (equity * MAX_ALLOC_FRACTION) - COMMISSION
    shares_alloc = max_alloc / friction_entry
    return int(min(shares_risk, shares_alloc))

def sector_cap_reached(open_positions, sector):
    count = sum(1 for p in open_positions.values() if p['sector'] == sector)
    return count >= MAX_SECTOR_POSITIONS

# ==============================================================================
# MODEL A - YANG (Trend Following, with trailing stops)
# ==============================================================================
def run_yang(ind, data, sp500, sp500_sma200, sp500_sma50):
    cash = START_CAPITAL
    equity = START_CAPITAL
    open_positions = {}
    trade_log = []
    equity_curve = []
    daily_returns = []

    dates = sp500.index[200:]

    for i in range(1, len(dates)):
        date = dates[i]
        prev_date = dates[i-1]
        cash = apply_cash_drag(cash)

        market_ok = (sp500.loc[prev_date] > sp500_sma200.loc[prev_date]) and \
                    (sp500.loc[prev_date] > sp500_sma50.loc[prev_date])

        just_exited_today = set()

        to_close = []
        for t, pos in open_positions.items():
            df = ind.get(t)
            if df is None or prev_date not in df.index:
                continue
            prev_close = df.loc[prev_date, 'Close']
            prev_sma50 = df.loc[prev_date, 'SMA50']
            y_atr = df.loc[prev_date, 'ATR14']

            calculated_trailing_stop = prev_close - (pos['stop_mult'] * y_atr)
            if calculated_trailing_stop > pos['stop']:
                pos['stop'] = calculated_trailing_stop

            if prev_close < prev_sma50 or prev_close <= pos['stop']:
                to_close.append(t)

        for t in to_close:
            pos = open_positions.pop(t)
            just_exited_today.add(t)
            if t not in data['Open'] or date not in data['Open'][t].index:
                continue
            open_price = data['Open'][t].loc[date]
            close_price = ind[t].loc[date, 'Close']
            exit_price = gap_down_exit(open_price, pos['stop'], close_price)
            exit_price = apply_exit_friction(exit_price)

            cash += pos['shares'] * exit_price
            cash -= COMMISSION

            pnl = (exit_price - pos['entry']) * pos['shares'] - COMMISSION
            trade_log.append({
                'ticker': t, 'entry_date': pos['entry_date'], 'exit_date': date,
                'entry_price': pos['entry'], 'exit_price': exit_price,
                'shares': pos['shares'], 'pnl': pnl
            })

        if market_ok:
            available_slots = MAX_TOTAL_POSITIONS - len(open_positions)
            candidates = []

            for t in tickers:
                if t in open_positions or t in just_exited_today:
                    continue
                df = ind.get(t)
                if df is None or prev_date not in df.index:
                    continue

                y_close = df.loc[prev_date, 'Close']
                y_sma50 = df.loc[prev_date, 'SMA50']
                y_prior_high = df.loc[prev_date, 'Prior20High']
                y_vol = df.loc[prev_date, 'Vol']
                y_avgvol30 = df.loc[prev_date, 'AvgVol30']
                y_atr = df.loc[prev_date, 'ATR14']
                y_sma20 = df.loc[prev_date, 'SMA20']

                sector = sectors.get(t, "Unknown")
                if sector_cap_reached(open_positions, sector):
                    continue

                trigger_entry = False
                mod_type = ""
                stop_mult = 2.0
                strength = 0.0

                if y_close > y_sma50 and y_vol > y_avgvol30 and y_close > y_prior_high:
                    trigger_entry = True
                    mod_type = "Yang Module A (High Vol Breakout)"
                    strength = (y_close - y_prior_high) / y_atr if y_atr > 0 else 0
                elif y_close > y_sma50 and y_vol < y_avgvol30 and y_close <= (y_sma20 + 2 * y_atr):
                    trigger_entry = True
                    mod_type = "Yang Module B (Low Vol Pullback)"
                    strength = (y_sma50 - y_close) / y_atr if y_atr > 0 else 0

                if trigger_entry:
                    candidates.append({
                        'ticker': t, 'mod_type': mod_type, 'stop_mult': stop_mult,
                        'atr': y_atr, 'strength': strength, 'sector': sector
                    })

            candidates.sort(key=lambda x: x['strength'], reverse=True)
            transient_equity = equity

            for c in candidates:
                if available_slots <= 0:
                    break
                t = c['ticker']
                sector = c['sector']
                if sector_cap_reached(open_positions, sector):
                    continue
                if t not in data['Open'] or date not in data['Open'][t].index:
                    continue

                raw_open = data['Open'][t].loc[date]
                stop = raw_open - (c['stop_mult'] * c['atr'])
                shares = position_size(raw_open, stop, transient_equity)
                entry_price = apply_entry_friction(raw_open)
                total_cost = (shares * entry_price) + COMMISSION

                if shares > 0 and cash >= total_cost:
                    cash -= total_cost
                    open_positions[t] = {
                        'entry': entry_price, 'stop': stop, 'shares': shares,
                        'sector': sector, 'entry_date': date,
                        'stop_mult': c['stop_mult'], 'days_held': 0
                    }
                    available_slots -= 1
                    transient_equity -= total_cost

        for t in open_positions:
            open_positions[t]['days_held'] = open_positions[t].get('days_held', 0) + 1

        open_val = 0.0
        for t, pos in open_positions.items():
            df = ind.get(t)
            if df is not None and date in df.index:
                open_val += pos['shares'] * df.loc[date, 'Close']
            else:
                open_val += pos['shares'] * pos['entry']

        prev_equity = equity
        equity = cash + open_val
        if i > 1:
            daily_returns.append((equity - prev_equity) / prev_equity)
        equity_curve.append(equity)

    return {
        'name': 'Yang', 'equity_curve': equity_curve, 'trade_log': trade_log,
        'daily_returns': daily_returns, 'dates': list(dates[1:])
    }

# ==============================================================================
# MODEL C - YIN (Mean Reversion)
# ==============================================================================
def run_yin(ind, data, sp500, sp500_sma200, sp500_sma50):
    cash = START_CAPITAL
    equity = START_CAPITAL
    open_positions = {}
    trade_log = []
    equity_curve = []
    daily_returns = []

    dates = sp500.index[200:]

    for i in range(1, len(dates)):
        date = dates[i]
        prev_date = dates[i-1]
        cash = apply_cash_drag(cash)

        market_ok = (sp500.loc[prev_date] > sp500_sma200.loc[prev_date]) and \
                    (sp500.loc[prev_date] > sp500_sma50.loc[prev_date])

        just_exited_today = set()

        to_close = []
        for t, pos in open_positions.items():
            df = ind.get(t)
            if df is None or prev_date not in df.index:
                continue
            prev_close = df.loc[prev_date, 'Close']
            prev_sma5 = df.loc[prev_date, 'SMA5']

            if pos['days_held'] >= 4:
                to_close.append(t)
            elif prev_close > prev_sma5 or prev_close <= pos['stop']:
                to_close.append(t)

        for t in to_close:
            pos = open_positions.pop(t)
            just_exited_today.add(t)
            if t not in data['Open'] or date not in data['Open'][t].index:
                continue
            open_price = data['Open'][t].loc[date]
            close_price = ind[t].loc[date, 'Close']
            exit_price = gap_down_exit(open_price, pos['stop'], close_price)
            exit_price = apply_exit_friction(exit_price)

            cash += pos['shares'] * exit_price
            cash -= COMMISSION

            pnl = (exit_price - pos['entry']) * pos['shares'] - COMMISSION
            trade_log.append({
                'ticker': t, 'entry_date': pos['entry_date'], 'exit_date': date,
                'entry_price': pos['entry'], 'exit_price': exit_price,
                'shares': pos['shares'], 'pnl': pnl
            })

        if market_ok:
            available_slots = MAX_TOTAL_POSITIONS - len(open_positions)
            candidates = []

            for t in tickers:
                if t in open_positions or t in just_exited_today:
                    continue
                df = ind.get(t)
                if df is None or prev_date not in df.index:
                    continue

                y_close = df.loc[prev_date, 'Close']
                y_rsi2 = df.loc[prev_date, 'RSI2']
                y_sma200 = df.loc[prev_date, 'SMA200']
                y_atr14 = df.loc[prev_date, 'ATR14']
                y_atr50 = df.loc[prev_date, 'ATR50']

                sector = sectors.get(t, "Unknown")
                if sector_cap_reached(open_positions, sector):
                    continue
                if pd.notna(y_atr50) and y_atr50 > 0 and y_atr14 > 2 * y_atr50:
                    continue

                trigger_entry = False
                mod_type = ""
                strength = 0.0

                if y_close > y_sma200 and y_rsi2 < 5:
                    trigger_entry = True
                    mod_type = "Yin Module B (Deep RSI < 5)"
                    strength = 15.0 - y_rsi2
                elif y_close > y_sma200 and y_rsi2 < 10:
                    trigger_entry = True
                    mod_type = "Yin Module A (Standard RSI < 10)"
                    strength = 10.0 - y_rsi2

                if trigger_entry:
                    candidates.append({
                        'ticker': t, 'mod_type': mod_type, 'atr': y_atr14,
                        'strength': strength, 'sector': sector
                    })

            candidates.sort(key=lambda x: x['strength'], reverse=True)
            transient_equity = equity

            for c in candidates:
                if available_slots <= 0:
                    break
                t = c['ticker']
                sector = c['sector']
                if sector_cap_reached(open_positions, sector):
                    continue
                if t not in data['Open'] or date not in data['Open'][t].index:
                    continue

                raw_open = data['Open'][t].loc[date]
                stop = raw_open - (1.0 * c['atr'])
                shares = position_size(raw_open, stop, transient_equity)
                entry_price = apply_entry_friction(raw_open)
                total_cost = (shares * entry_price) + COMMISSION

                if shares > 0 and cash >= total_cost:
                    cash -= total_cost
                    open_positions[t] = {
                        'entry': entry_price, 'stop': stop, 'shares': shares,
                        'sector': sector, 'entry_date': date, 'days_held': 0
                    }
                    available_slots -= 1
                    transient_equity -= total_cost

        for t in open_positions:
            open_positions[t]['days_held'] += 1

        open_val = 0.0
        for t, pos in open_positions.items():
            df = ind.get(t)
            if df is not None and date in df.index:
                open_val += pos['shares'] * df.loc[date, 'Close']
            else:
                open_val += pos['shares'] * pos['entry']

        prev_equity = equity
        equity = cash + open_val
        if i > 1:
            daily_returns.append((equity - prev_equity) / prev_equity)
        equity_curve.append(equity)

    return {
        'name': 'Yin', 'equity_curve': equity_curve, 'trade_log': trade_log,
        'daily_returns': daily_returns, 'dates': list(dates[1:])
    }

# ==============================================================================
# MODEL B - ZHAN (Volatility Expansion, with trailing stops)
# ==============================================================================
def run_zhan(ind, data, sp500, sp500_sma200, sp500_sma50):
    cash = START_CAPITAL
    equity = START_CAPITAL
    open_positions = {}
    trade_log = []
    equity_curve = []
    daily_returns = []

    dates = sp500.index[200:]

    for i in range(1, len(dates)):
        date = dates[i]
        prev_date = dates[i-1]
        cash = apply_cash_drag(cash)

        market_ok = (sp500.loc[prev_date] > sp500_sma200.loc[prev_date]) and \
                    (sp500.loc[prev_date] > sp500_sma50.loc[prev_date])

        just_exited_today = set()

        to_close = []
        for t, pos in open_positions.items():
            df = ind.get(t)
            if df is None or prev_date not in df.index:
                continue
            prev_close = df.loc[prev_date, 'Close']
            prev_kc_lower = df.loc[prev_date, 'KC_LOWER']
            y_atr = df.loc[prev_date, 'ATR14']

            calculated_trailing_stop = prev_close - (pos['stop_mult'] * y_atr)
            if calculated_trailing_stop > pos['stop']:
                pos['stop'] = calculated_trailing_stop

            if prev_close < prev_kc_lower or prev_close <= pos['stop']:
                to_close.append(t)

        for t in to_close:
            pos = open_positions.pop(t)
            just_exited_today.add(t)
            if t not in data['Open'] or date not in data['Open'][t].index:
                continue
            open_price = data['Open'][t].loc[date]
            close_price = ind[t].loc[date, 'Close']
            exit_price = gap_down_exit(open_price, pos['stop'], close_price)
            exit_price = apply_exit_friction(exit_price)

            cash += pos['shares'] * exit_price
            cash -= COMMISSION

            pnl = (exit_price - pos['entry']) * pos['shares'] - COMMISSION
            trade_log.append({
                'ticker': t, 'entry_date': pos['entry_date'], 'exit_date': date,
                'entry_price': pos['entry'], 'exit_price': exit_price,
                'shares': pos['shares'], 'pnl': pnl
            })

        if market_ok:
            available_slots = MAX_TOTAL_POSITIONS - len(open_positions)
            candidates = []

            for t in tickers:
                if t in open_positions or t in just_exited_today:
                    continue
                df = ind.get(t)
                if df is None or prev_date not in df.index:
                    continue
                curr_idx = df.index.get_loc(prev_date)
                if curr_idx < 4:
                    continue

                y_close = df.loc[prev_date, 'Close']
                y_ema20 = df.loc[prev_date, 'EMA20']
                y_kc_upper = df.loc[prev_date, 'KC_UPPER']
                y_vol = df.loc[prev_date, 'Vol']
                y_avgvol30 = df.loc[prev_date, 'AvgVol30']
                y_atr = df.loc[prev_date, 'ATR14']

                sector = sectors.get(t, "Unknown")
                if sector_cap_reached(open_positions, sector):
                    continue

                cons = (
                    df.iloc[curr_idx-1]['Close'] < df.iloc[curr_idx-1]['KC_UPPER'] and
                    df.iloc[curr_idx-1]['Close'] > df.iloc[curr_idx-1]['KC_LOWER'] and
                    df.iloc[curr_idx-2]['Close'] < df.iloc[curr_idx-2]['KC_UPPER'] and
                    df.iloc[curr_idx-2]['Close'] > df.iloc[curr_idx-2]['KC_LOWER'] and
                    df.iloc[curr_idx-3]['Close'] < df.iloc[curr_idx-3]['KC_UPPER'] and
                    df.iloc[curr_idx-3]['Close'] > df.iloc[curr_idx-3]['KC_LOWER']
                )

                trigger_entry = False
                mod_type = ""
                strength = 0.0
                stop_mult = 1.5

                if y_close > y_kc_upper and y_vol > y_avgvol30 and cons:
                    trigger_entry = True
                    mod_type = "Zhan Module A (Keltner Breakout)"
                    strength = (y_close - y_kc_upper) / y_atr if y_atr > 0 else 0
                elif y_close > y_ema20 and y_vol < y_avgvol30 and cons:
                    trigger_entry = True
                    mod_type = "Zhan Module B (EMA Breakout)"
                    strength = (y_close - y_ema20) / y_atr if y_atr > 0 else 0

                if trigger_entry:
                    candidates.append({
                        'ticker': t, 'mod_type': mod_type, 'atr': y_atr,
                        'strength': strength, 'sector': sector, 'stop_mult': stop_mult
                    })

            candidates.sort(key=lambda x: x['strength'], reverse=True)
            transient_equity = equity

            for c in candidates:
                if available_slots <= 0:
                    break
                t = c['ticker']
                sector = c['sector']
                if sector_cap_reached(open_positions, sector):
                    continue
                if t not in data['Open'] or date not in data['Open'][t].index:
                    continue

                raw_open = data['Open'][t].loc[date]
                stop = raw_open - (c['stop_mult'] * c['atr'])
                shares = position_size(raw_open, stop, transient_equity)
                entry_price = apply_entry_friction(raw_open)
                total_cost = (shares * entry_price) + COMMISSION

                if shares > 0 and cash >= total_cost:
                    cash -= total_cost
                    open_positions[t] = {
                        'entry': entry_price, 'stop': stop, 'shares': shares,
                        'sector': sector, 'entry_date': date,
                        'stop_mult': c['stop_mult'], 'days_held': 0
                    }
                    available_slots -= 1
                    transient_equity -= total_cost

        for t in open_positions:
            open_positions[t]['days_held'] = open_positions[t].get('days_held', 0) + 1

        open_val = 0.0
        for t, pos in open_positions.items():
            df = ind.get(t)
            if df is not None and date in df.index:
                open_val += pos['shares'] * df.loc[date, 'Close']
            else:
                open_val += pos['shares'] * pos['entry']

        prev_equity = equity
        equity = cash + open_val
        if i > 1:
            daily_returns.append((equity - prev_equity) / prev_equity)
        equity_curve.append(equity)

    return {
        'name': 'Zhan', 'equity_curve': equity_curve, 'trade_log': trade_log,
        'daily_returns': daily_returns, 'dates': list(dates[1:])
    }

# ==============================================================================
# ENSEMBLE - dynamic Yin/Yang/Zhan blend with regime + performance weighting
# ==============================================================================
def rolling_stats(series, window=30):
    s = pd.Series(series)
    if len(s) < window:
        return {'mean': 0.0, 'std': 0.0}
    roll = s.iloc[-window:]
    return {'mean': roll.mean(), 'std': roll.std()}

def compute_strategy_metrics(results_dict):
    metrics = {}
    for name, res in results_dict.items():
        daily = res.get('daily_returns', [])
        eq = res.get('equity_curve', [])
        if len(daily) == 0 or len(eq) == 0:
            metrics[name] = {'sharpe_30d': 0.0, 'dd': 0.0}
            continue
        rs = rolling_stats(daily, window=30)
        sharpe_30d = ((rs['mean'] - DAILY_RF) / rs['std']) * np.sqrt(252) if rs['std'] > 0 else 0.0
        eq_series = pd.Series(eq)
        peak = eq_series.cummax()
        dd_series = (eq_series - peak) / peak
        dd = dd_series.iloc[-1] if len(dd_series) > 0 else 0.0
        metrics[name] = {'sharpe_30d': sharpe_30d, 'dd': dd}
    return metrics

def build_strategy_weights(metrics):
    raw = {}
    for name, m in metrics.items():
        sharpe = max(m['sharpe_30d'], 0.0)
        dd = m['dd']
        if dd < -0.2:
            dd_mult = 0.1
        elif dd < -0.1:
            dd_mult = 0.5
        else:
            dd_mult = 1.0
        raw[name] = sharpe * dd_mult
    total = sum(raw.values())
    if total <= 0:
        return {name: 1/3 for name in raw.keys()}
    return {name: v / total for name, v in raw.items()}

def compute_vol_scale(portfolio_daily_returns, target_vol=0.10, window=30):
    rs = rolling_stats(portfolio_daily_returns, window=window)
    if rs['std'] <= 0:
        return 1.0
    current_annual_vol = rs['std'] * np.sqrt(252)
    if current_annual_vol <= 0:
        return 1.0
    return target_vol / current_annual_vol

def detect_vol_regime(sp500, dates, window=20):
    if len(dates) < window + 1:
        return 'medium'
    closes = sp500.loc[dates[-window:]]
    returns = closes.pct_change().dropna()
    if len(returns) == 0:
        return 'medium'
    vol = returns.std() * np.sqrt(252)
    if vol < 0.12:
        return 'low'
    elif vol < 0.20:
        return 'medium'
    else:
        return 'high'

def regime_strategy_weights(regime):
    if regime == 'low':
        return {'Yang': 1.4, 'Yin': 1.0, 'Zhan': 0.8}
    elif regime == 'medium':
        return {'Yang': 1.0, 'Yin': 1.0, 'Zhan': 1.0}
    else:
        return {'Yang': 0.7, 'Yin': 1.2, 'Zhan': 1.3}

def run_ensemble_portfolio_advanced(ind, data, sp500, sp500_sma200, sp500_sma50, config_params):
    """Multi-strategy ensemble with volatility targeting, regime-aware
    weighting, and performance-based (Sharpe + drawdown) reweighting.

    FIXED: strategy_results used to be initialized empty and never updated,
    which silently forced compute_strategy_metrics -> build_strategy_weights
    to fall back to flat 1/3 weights every single day, regardless of which
    strategy was actually performing better. Now each strategy's realized
    trade PnL is tracked as it happens and converted into a running proxy
    equity curve, so the Sharpe/drawdown-based weighting actually has real
    data to react to. This proxy only moves on days a trade in that
    strategy closes (not full mark-to-market every day) -- a reasonable
    approximation for weighting purposes, not a full attribution system.
    """
    cash = START_CAPITAL
    equity = START_CAPITAL
    open_positions = {}
    trade_log = []
    equity_curve = []
    daily_returns = []

    dates = sp500.index[200:]
    portfolio_daily = []

    strategy_names = ['Yin', 'Yang', 'Zhan']
    strategy_cum_pnl = {name: 0.0 for name in strategy_names}
    strategy_results = {
        name: {'daily_returns': [], 'equity_curve': [START_CAPITAL / 3]}
        for name in strategy_names
    }

    for i in range(1, len(dates)):
        date = dates[i]
        prev_date = dates[i-1]
        cash = apply_cash_drag(cash)

        market_ok = (sp500.loc[prev_date] > sp500_sma200.loc[prev_date]) and \
                    (sp500.loc[prev_date] > sp500_sma50.loc[prev_date])

        just_exited_today = set()

        to_close = []
        for t, pos in open_positions.items():
            df = ind.get(t)
            if df is None or prev_date not in df.index:
                continue
            prev_close = df.loc[prev_date, 'Close']
            y_atr14 = df.loc[prev_date, 'ATR14']
            strategy = pos['strategy']

            if strategy == 'Yin':
                prev_sma5 = df.loc[prev_date, 'SMA5']
                if pos['days_held'] >= 4 or prev_close > prev_sma5 or prev_close <= pos['stop']:
                    to_close.append(t)
            elif strategy == 'Yang':
                prev_sma50 = df.loc[prev_date, 'SMA50']
                calc_stop = prev_close - (pos['stop_mult'] * y_atr14)
                if calc_stop > pos['stop']:
                    pos['stop'] = calc_stop
                if prev_close < prev_sma50 or prev_close <= pos['stop']:
                    to_close.append(t)
            elif strategy == 'Zhan':
                prev_kc_lower = df.loc[prev_date, 'KC_LOWER']
                calc_stop = prev_close - (pos['stop_mult'] * y_atr14)
                if calc_stop > pos['stop']:
                    pos['stop'] = calc_stop
                if prev_close < prev_kc_lower or prev_close <= pos['stop']:
                    to_close.append(t)

        for t in to_close:
            pos = open_positions.pop(t)
            just_exited_today.add(t)
            if t not in data['Open'] or date not in data['Open'][t].index:
                continue
            open_price = data['Open'][t].loc[date]
            close_price = ind[t].loc[date, 'Close']
            exit_price = gap_down_exit(open_price, pos['stop'], close_price)
            exit_price = apply_exit_friction(exit_price)

            cash += pos['shares'] * exit_price
            cash -= COMMISSION

            pnl = (exit_price - pos['entry']) * pos['shares'] - COMMISSION
            trade_log.append({
                'ticker': t, 'entry_date': pos['entry_date'], 'exit_date': date,
                'entry_price': pos['entry'], 'exit_price': exit_price,
                'shares': pos['shares'], 'pnl': pnl, 'strategy': pos['strategy']
            })
            strategy_cum_pnl[pos['strategy']] += pnl

        for name in strategy_names:
            proxy_equity = (START_CAPITAL / 3) + strategy_cum_pnl[name]
            prev_proxy = strategy_results[name]['equity_curve'][-1]
            strategy_results[name]['equity_curve'].append(proxy_equity)
            if prev_proxy != 0:
                strategy_results[name]['daily_returns'].append((proxy_equity - prev_proxy) / prev_proxy)

        metrics = compute_strategy_metrics(strategy_results)
        base_weights = build_strategy_weights(metrics)
        regime = detect_vol_regime(sp500, dates[:i+1])
        regime_mult = regime_strategy_weights(regime)
        strategy_weight = {
            name: base_weights.get(name, 0.0) * regime_mult.get(name, 1.0)
            for name in strategy_names
        }
        vol_scale = compute_vol_scale(portfolio_daily, target_vol=config_params.get('target_vol', 0.10))

        if market_ok:
            available_slots = config_params['max_total_positions'] - len(open_positions)
            candidates = []

            for t in tickers:
                if t in open_positions or t in just_exited_today:
                    continue
                df = ind.get(t)
                if df is None or prev_date not in df.index:
                    continue

                sector = sectors.get(t, "Unknown")
                if sector_cap_reached(open_positions, sector):
                    continue

                y_close = df.loc[prev_date, 'Close']
                y_vol = df.loc[prev_date, 'Vol']
                y_avgvol30 = df.loc[prev_date, 'AvgVol30']
                y_atr14 = df.loc[prev_date, 'ATR14']

                y_rsi2 = df.loc[prev_date, 'RSI2']
                y_sma200 = df.loc[prev_date, 'SMA200']
                y_atr50 = df.loc[prev_date, 'ATR50']
                yin_vol_ok = not (pd.notna(y_atr50) and y_atr50 > 0 and y_atr14 > 2 * y_atr50)

                if yin_vol_ok and y_close > y_sma200 and y_rsi2 < 10:
                    strength = (15.0 - y_rsi2) if y_rsi2 < 5 else (10.0 - y_rsi2)
                    strength *= strategy_weight['Yin']
                    candidates.append({'ticker': t, 'strategy': 'Yin', 'atr': y_atr14,
                                        'sector': sector, 'stop_mult': 1.0, 'strength': strength})
                    continue

                y_sma50 = df.loc[prev_date, 'SMA50']
                y_prior_high = df.loc[prev_date, 'Prior20High']
                y_sma20 = df.loc[prev_date, 'SMA20']

                if y_close > y_sma50 and y_vol > y_avgvol30 and y_close > y_prior_high:
                    base_strength = (y_close - y_prior_high) / y_atr14 if y_atr14 > 0 else 0
                    strength = base_strength * strategy_weight['Yang']
                    candidates.append({'ticker': t, 'strategy': 'Yang', 'atr': y_atr14,
                                        'sector': sector, 'stop_mult': 2.0, 'strength': strength})
                    continue
                elif y_close > y_sma50 and y_vol < y_avgvol30 and y_close <= (y_sma20 + 2 * y_atr14):
                    base_strength = (y_sma50 - y_close) / y_atr14 if y_atr14 > 0 else 0
                    strength = base_strength * strategy_weight['Yang']
                    candidates.append({'ticker': t, 'strategy': 'Yang', 'atr': y_atr14,
                                        'sector': sector, 'stop_mult': 2.0, 'strength': strength})
                    continue

                try:
                    curr_idx = df.index.get_loc(prev_date)
                except KeyError:
                    continue
                if curr_idx < 4:
                    continue

                y_ema20 = df.loc[prev_date, 'EMA20']
                y_kc_upper = df.loc[prev_date, 'KC_UPPER']
                cons = (
                    df.iloc[curr_idx-1]['Close'] < df.iloc[curr_idx-1]['KC_UPPER'] and
                    df.iloc[curr_idx-1]['Close'] > df.iloc[curr_idx-1]['KC_LOWER'] and
                    df.iloc[curr_idx-2]['Close'] < df.iloc[curr_idx-2]['KC_UPPER'] and
                    df.iloc[curr_idx-2]['Close'] > df.iloc[curr_idx-2]['KC_LOWER'] and
                    df.iloc[curr_idx-3]['Close'] < df.iloc[curr_idx-3]['KC_UPPER'] and
                    df.iloc[curr_idx-3]['Close'] > df.iloc[curr_idx-3]['KC_LOWER']
                )

                if y_close > y_kc_upper and y_vol > y_avgvol30 and cons:
                    base_strength = (y_close - y_kc_upper) / y_atr14 if y_atr14 > 0 else 0
                    strength = base_strength * strategy_weight['Zhan']
                    candidates.append({'ticker': t, 'strategy': 'Zhan', 'atr': y_atr14,
                                        'sector': sector, 'stop_mult': 1.5, 'strength': strength})
                elif y_close > y_ema20 and y_vol < y_avgvol30 and cons:
                    base_strength = (y_close - y_ema20) / y_atr14 if y_atr14 > 0 else 0
                    strength = base_strength * strategy_weight['Zhan']
                    candidates.append({'ticker': t, 'strategy': 'Zhan', 'atr': y_atr14,
                                        'sector': sector, 'stop_mult': 1.5, 'strength': strength})

            candidates.sort(key=lambda x: x['strength'], reverse=True)
            transient_equity = equity

            for c in candidates:
                if available_slots <= 0:
                    break
                t = c['ticker']
                sector = c['sector']
                if sector_cap_reached(open_positions, sector):
                    continue
                if t not in data['Open'] or date not in data['Open'][t].index:
                    continue

                raw_open = data['Open'][t].loc[date]
                stop = raw_open - (c['stop_mult'] * c['atr'])
                base_shares = position_size(raw_open, stop, transient_equity)
                shares = int(base_shares * vol_scale)
                entry_price = apply_entry_friction(raw_open)
                total_cost = (shares * entry_price) + COMMISSION

                if shares > 0 and cash >= total_cost:
                    cash -= total_cost
                    open_positions[t] = {
                        'entry': entry_price, 'stop': stop, 'shares': shares,
                        'sector': sector, 'entry_date': date, 'strategy': c['strategy'],
                        'stop_mult': c['stop_mult'], 'days_held': 0
                    }
                    available_slots -= 1
                    transient_equity -= total_cost

        for t in open_positions:
            open_positions[t]['days_held'] += 1

        open_val = 0.0
        for t, pos in open_positions.items():
            df = ind.get(t)
            if df is not None and date in df.index:
                open_val += pos['shares'] * df.loc[date, 'Close']
            else:
                open_val += pos['shares'] * pos['entry']

        prev_equity = equity
        equity = cash + open_val
        if i > 1:
            ret = (equity - prev_equity) / prev_equity
            daily_returns.append(ret)
            portfolio_daily.append(ret)
        equity_curve.append(equity)

    return {
        'name': 'Ensemble', 'equity_curve': equity_curve, 'trade_log': trade_log,
        'daily_returns': daily_returns, 'dates': list(dates[1:])
    }

def run_ensemble_suite(ind, data, sp500, sp500_sma200, sp500_sma50, config_params):
    """Explicit wrapper: temporarily sets MAX_TOTAL_POSITIONS / MAX_ALLOC_FRACTION
    / MAX_SECTOR_POSITIONS from config_params (instead of silently relying on
    whatever the module-level globals happened to be left at from a prior
    run), then restores them. Mirrors run_full_suite's pattern."""
    global MAX_TOTAL_POSITIONS, MAX_ALLOC_FRACTION, MAX_SECTOR_POSITIONS
    orig = (MAX_TOTAL_POSITIONS, MAX_ALLOC_FRACTION, MAX_SECTOR_POSITIONS)
    MAX_TOTAL_POSITIONS = config_params.get('max_total_positions', 8)
    MAX_ALLOC_FRACTION = config_params.get('max_alloc_fraction', 1/6)
    MAX_SECTOR_POSITIONS = config_params.get('max_sector_positions', 2)

    result = run_ensemble_portfolio_advanced(ind, data, sp500, sp500_sma200, sp500_sma50, config_params)

    MAX_TOTAL_POSITIONS, MAX_ALLOC_FRACTION, MAX_SECTOR_POSITIONS = orig
    return result

# ==========================================
# BENCHMARK - SPX BUY & HOLD
# ==========================================
def run_benchmark(sp500, dates):
    """Buy-and-hold the index over the same dates the strategies traded.
    Uses dates[1:] to match the strategy loops (which start at i=1), so
    equity_curve/daily_returns lengths line up with every other model."""
    first_date = dates[1]
    base_price = sp500.loc[first_date]
    equity_curve = []
    daily_returns = []
    prev_equity = START_CAPITAL

    for i in range(1, len(dates)):
        date = dates[i]
        equity = START_CAPITAL * (sp500.loc[date] / base_price)
        if i > 1:
            daily_returns.append((equity - prev_equity) / prev_equity)
        equity_curve.append(equity)
        prev_equity = equity

    return {
        'name': 'SPX Buy & Hold', 'equity_curve': equity_curve, 'trade_log': [],
        'daily_returns': daily_returns, 'dates': list(dates[1:])
    }

# ==============================================================================
# PERFORMANCE METRICS
# ==============================================================================
def compute_metrics(model):
    eq = pd.Series(model['equity_curve'])
    daily = np.array(model['daily_returns'])
    trades = model.get('trade_log', [])

    running_peak = eq.cummax()
    drawdown = (eq - running_peak) / running_peak if not running_peak.empty else eq
    max_dd = drawdown.min() * 100 if not drawdown.empty else 0

    total_return = (eq.iloc[-1] - START_CAPITAL) / START_CAPITAL * 100 if not eq.empty else 0

    daily_std = np.std(daily)
    sharpe = ((np.mean(daily) - DAILY_RF) / daily_std) * np.sqrt(252) if daily_std > 0 else 0

    downside_returns = np.where(daily < 0, daily, 0)
    downside_dev = np.std(downside_returns)
    sortino = ((np.mean(daily) - DAILY_RF) / downside_dev) * np.sqrt(252) if downside_dev > 0 else 0

    for tr in trades:
        tr['pct_return'] = (tr['exit_price'] - tr['entry_price']) / tr['entry_price']

    gross_pct_profit = sum(tr['pct_return'] for tr in trades if tr['pct_return'] > 0)
    gross_pct_loss = -sum(tr['pct_return'] for tr in trades if tr['pct_return'] < 0)
    profit_factor = gross_pct_profit / gross_pct_loss if gross_pct_loss > 0 else (0.0 if total_return <= 0 else np.nan)

    wins = sum(1 for tr in trades if tr['pct_return'] > 0)
    total_trades = len(trades)
    win_rate = (wins / total_trades * 100) if total_trades > 0 else np.nan

    return {
        'name': model['name'], 'total_return': total_return, 'max_dd': max_dd,
        'sharpe': sharpe, 'sortino': sortino, 'profit_factor': profit_factor,
        'win_rate': win_rate, 'total_trades': total_trades,
        'equity_curve': eq, 'drawdown': drawdown, 'trade_log': trades
    }

def compare_models(results, SHOW_CHARTS=False):
    metrics = [compute_metrics(r) for r in results]

    if SHOW_CHARTS:
        plt.figure(figsize=(12, 6))
        for m in metrics:
            plt.plot(m['equity_curve'], label=m['name'])
        plt.title("Equity Curves")
        plt.legend()
        plt.grid(True)
        plt.show()

    table = pd.DataFrame({
        'Model': [m['name'] for m in metrics],
        'Total Return %': [m['total_return'] for m in metrics],
        'Max Drawdown %': [m['max_dd'] for m in metrics],
        'Sharpe': [m['sharpe'] for m in metrics],
        'Sortino': [m['sortino'] for m in metrics],
        'Profit Factor': [m['profit_factor'] for m in metrics],
        'Win Rate %': [m['win_rate'] for m in metrics],
        'Trades': [m['total_trades'] for m in metrics]
    })

    print("\n==============================================================")
    print("MODEL PERFORMANCE COMPARISON (including SPX benchmark)")
    print("==============================================================")
    print(table.to_string(index=False))

    print("\n==============================================================")
    print("STRATEGY RANKING (by Total Return)")
    print("==============================================================")
    ranked = table.sort_values(by='Total Return %', ascending=False)
    print(ranked[['Model', 'Total Return %']].to_string(index=False))

    bench_row = table[table['Model'] == 'SPX Buy & Hold']
    if not bench_row.empty:
        bench_return = bench_row['Total Return %'].values[0]
        bench_sharpe = bench_row['Sharpe'].values[0]
        print("\n==============================================================")
        print("ALPHA OVER SPX BUY & HOLD")
        print("==============================================================")
        for m in metrics:
            if m['name'] == 'SPX Buy & Hold':
                continue
            excess_return = m['total_return'] - bench_return
            excess_sharpe = m['sharpe'] - bench_sharpe
            print(f"{m['name']:<30}: {excess_return:+7.2f}% return vs SPX | {excess_sharpe:+.2f} Sharpe vs SPX")

    for m in metrics:
        if not m['trade_log']:
            continue
        df = pd.DataFrame(m['trade_log'])
        csv_name = f"{m['name'].lower().replace(' ', '_').replace('(', '').replace(')', '')}_trades.csv"
        df.to_csv(csv_name, index=False)
        print(f"Exported {m['name']} trades -> {csv_name}")

# ==============================================================================
# MONTE CARLO - vectorized block bootstrap
# ==============================================================================
def monte_carlo_bootstrap(daily_returns, n_sims=5000, block_size=5, seed=42):
    rng = np.random.default_rng(seed)
    daily = np.asarray(daily_returns, dtype=float)
    n = len(daily)
    if n == 0:
        return None
    block_size = min(block_size, n)

    num_blocks_needed = int(np.ceil(n / block_size))
    max_start_idx = n - block_size + 1 if n > block_size else 1

    start_indices = rng.integers(0, max_start_idx, size=(n_sims, num_blocks_needed))
    sim_paths = np.zeros((n_sims, num_blocks_needed * block_size))
    for block_offset in range(block_size):
        sim_paths[:, block_offset::block_size] = daily[start_indices + block_offset]
    sim_paths = sim_paths[:, :n]

    eq_paths = START_CAPITAL * np.cumprod(1 + sim_paths, axis=1)
    running_peaks = np.maximum.accumulate(eq_paths, axis=1)
    drawdowns = (eq_paths - running_peaks) / running_peaks
    max_dds = np.min(drawdowns, axis=1) * 100
    total_returns = (eq_paths[:, -1] - START_CAPITAL) / START_CAPITAL * 100

    means = np.mean(sim_paths - DAILY_RF, axis=1)
    stds = np.std(sim_paths, axis=1)
    sharpes = np.where(stds > 0, (means / stds) * np.sqrt(252), 0.0)

    return {'total_return': total_returns, 'max_dd': max_dds, 'sharpe': sharpes}

def run_monte_carlo_suite(results, n_sims=5000, block_size=5, seed=42):
    summaries = []
    mc_by_name = {}

    for r in results:
        actual_total_return = (pd.Series(r['equity_curve']).iloc[-1] - START_CAPITAL) / START_CAPITAL * 100
        mc = monte_carlo_bootstrap(r['daily_returns'], n_sims=n_sims, block_size=block_size, seed=seed)
        if mc is None:
            continue
        mc_by_name[r['name']] = mc
        summaries.append({
            'name': r['name'],
            'return_p5': np.percentile(mc['total_return'], 5),
            'return_p25': np.percentile(mc['total_return'], 25),
            'return_median': np.percentile(mc['total_return'], 50),
            'return_p75': np.percentile(mc['total_return'], 75),
            'return_p95': np.percentile(mc['total_return'], 95),
            'dd_p5': np.percentile(mc['max_dd'], 5),
            'sharpe_median': np.percentile(mc['sharpe'], 50),
            'pct_sims_positive': (mc['total_return'] > 0).mean() * 100,
            'actual_return': actual_total_return,
        })

    table = pd.DataFrame(summaries)
    print("\n==============================================================")
    print(f"MONTE CARLO (block bootstrap, {n_sims} sims, block size {block_size} days)")
    print("==============================================================")
    print_cols = ['name', 'return_p5', 'return_p25', 'return_median', 'return_p75',
                  'return_p95', 'dd_p5', 'sharpe_median', 'pct_sims_positive', 'actual_return']
    rename = {
        'name': 'Model', 'return_p5': 'Return P5%', 'return_p25': 'Return P25%',
        'return_median': 'Return Median%', 'return_p75': 'Return P75%',
        'return_p95': 'Return P95%', 'dd_p5': 'Tail Risk DD% (P5)',
        'sharpe_median': 'Median Sharpe', 'pct_sims_positive': '% Sims Profitable',
        'actual_return': 'Actual Return%'
    }
    print(table[print_cols].rename(columns=rename).round(2).to_string(index=False))

    if 'SPX Buy & Hold' in mc_by_name:
        bench_tr = mc_by_name['SPX Buy & Hold']['total_return']
        print("\n==============================================================")
        print("P(STRATEGY PATH BEATS SPX PATH) -- independently resampled Monte Carlo runs")
        print("==============================================================")
        for name, mc in mc_by_name.items():
            if name == 'SPX Buy & Hold':
                continue
            n = min(len(mc['total_return']), len(bench_tr))
            win_rate = (mc['total_return'][:n] > bench_tr[:n]).mean() * 100
            print(f"{name:>10}: beats SPX in {win_rate:.1f}% of paired simulations")

    return table

# ==============================================================================
# CONSTRAINT SWEEP - Manual vs Unconstrained
# ==============================================================================
def run_full_suite(ind, data, sp500, sp500_sma200, sp500_sma50,
                    max_total_positions, max_alloc_fraction, max_sector_positions,
                    label=""):
    """Runs Yang/Zhan/Yin with the given position-limit parameters, mutating
    the module-level globals for the duration of the run then restoring
    them."""
    global MAX_TOTAL_POSITIONS, MAX_ALLOC_FRACTION, MAX_SECTOR_POSITIONS
    orig = (MAX_TOTAL_POSITIONS, MAX_ALLOC_FRACTION, MAX_SECTOR_POSITIONS)
    MAX_TOTAL_POSITIONS = max_total_positions
    MAX_ALLOC_FRACTION = max_alloc_fraction
    MAX_SECTOR_POSITIONS = max_sector_positions

    suffix = f" {label}" if label else ""
    yang = run_yang(ind, data, sp500, sp500_sma200, sp500_sma50)
    zhan = run_zhan(ind, data, sp500, sp500_sma200, sp500_sma50)
    yin = run_yin(ind, data, sp500, sp500_sma200, sp500_sma50)
    yang['name'] += suffix
    zhan['name'] += suffix
    yin['name'] += suffix

    MAX_TOTAL_POSITIONS, MAX_ALLOC_FRACTION, MAX_SECTOR_POSITIONS = orig
    return yang, zhan, yin

# ==============================================================================
# MAIN
# ==============================================================================
if __name__ == "__main__":
    data = load_data()
    ind, sp500, sp500_sma200, sp500_sma50 = build_indicators(data)
    dates = sp500.index[200:]

    print("\n### RUN 1: Manual-Tracking Limits (original) ###")
    print("    MAX_TOTAL_POSITIONS=6, MAX_ALLOC_FRACTION=1/6, MAX_SECTOR_POSITIONS=2")
    yang_manual, zhan_manual, yin_manual = run_full_suite(
        ind, data, sp500, sp500_sma200, sp500_sma50,
        max_total_positions=6, max_alloc_fraction=1/6, max_sector_positions=2,
        label="(Manual Limits)"
    )

    print("\n### RUN 2: Unconstrained (no manual-tracking caps) ###")
    print(f"    MAX_TOTAL_POSITIONS={len(tickers)}, MAX_ALLOC_FRACTION=1.0, "
          f"MAX_SECTOR_POSITIONS={len(tickers)}")
    yang_free, zhan_free, yin_free = run_full_suite(
        ind, data, sp500, sp500_sma200, sp500_sma50,
        max_total_positions=len(tickers), max_alloc_fraction=1.0, max_sector_positions=len(tickers),
        label="(Unconstrained)"
    )

    print("\n### RUN 3: Ensemble (dynamic Yin/Yang/Zhan blend) ###")
    ensemble_results = run_ensemble_suite(
        ind, data, sp500, sp500_sma200, sp500_sma50,
        config_params={'max_total_positions': 8, 'max_alloc_fraction': 1/6,
                        'max_sector_positions': 2, 'target_vol': 0.10}
    )

    print("\nProcessing SPX Buy & Hold benchmark...")
    benchmark_results = run_benchmark(sp500, dates)

    all_results = [
        yang_manual, zhan_manual, yin_manual,
        yang_free, zhan_free, yin_free,
        ensemble_results,
        benchmark_results
    ]

    compare_models(all_results, SHOW_CHARTS=False)

    print("\nRunning Monte Carlo block-bootstrap analysis...")
    run_monte_carlo_suite(all_results, n_sims=5000, block_size=5, seed=42)

#added the ensemble(basically a collection of the 3 individual models) 
