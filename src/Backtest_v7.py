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

tickers = ['MSFT','AMZN','JPM','UNH','XOM','WMT','NVDA','AMD','META','TSLA']
sectors = {
    'MSFT':'Tech','NVDA':'Tech','AMD':'Tech','META':'Tech',
    'AMZN':'Consumer','TSLA':'Consumer','WMT':'Consumer',
    'JPM':'Finance','UNH':'Healthcare','XOM':'Energy'
}

# ==========================================
# LOAD DATA
# ==========================================
def load_data():
    print("Downloading historical data...")
    # auto_adjust pinned explicitly (matches the live/forward-test script) so
    # backtest and live signals are computed off the SAME price basis (raw,
    # unadjusted close). Leaving this to yfinance's default risked the
    # backtest silently running on split/dividend-adjusted prices while live
    # trading ran on raw prices -- two different price series for names like
    # NVDA/TSLA/AMZN that have had splits in this window.
    data = yf.download(tickers + ['^GSPC'], start="2023-01-01", end="2026-07-03",
                        auto_adjust=False)
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
        df = pd.DataFrame(index=close.index)
        df['Close'] = close[t]
        df['High'] = high[t]
        df['Low'] = low[t]
        df['Vol'] = vol[t]

        # Trend indicators
        df['SMA50'] = df['Close'].rolling(50).mean()
        df['SMA20'] = df['Close'].rolling(20).mean()
        df['EMA20'] = df['Close'].ewm(span=20).mean()
        df['SMA5'] = df['Close'].rolling(5).mean()
        df['SMA200'] = df['Close'].rolling(200).mean()

        # Volume indicators
        df['AvgVol30'] = df['Vol'].rolling(30).mean()

        # Breakout indicators
        df['Prior20High'] = df['Close'].rolling(20).max().shift(1)

        # ATR(14) and ATR(50) for volatility breaker
        tr1 = df['High'] - df['Low']
        tr2 = (df['High'] - df['Close'].shift(1)).abs()
        tr3 = (df['Low'] - df['Close'].shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df['ATR14'] = tr.rolling(14).mean()
        df['ATR50'] = tr.rolling(50).mean()

        # RSI(2)
        delta = df['Close'].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(2).mean()
        avg_loss = loss.rolling(2).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        df['RSI2'] = 100 - (100 / (1 + rs))

        # Keltner Channels (1.5x ATR)
        df['KC_MID'] = df['EMA20']
        df['KC_UPPER'] = df['EMA20'] + 1.5 * df['ATR14']
        df['KC_LOWER'] = df['EMA20'] - 1.5 * df['ATR14']

        ind[t] = df

    # Market filter: 200-SMA and 50-SMA
    sp500 = close['^GSPC']
    sp500_sma200 = sp500.rolling(200).mean()
    sp500_sma50 = sp500.rolling(50).mean()

    return ind, sp500, sp500_sma200, sp500_sma50

# ==========================================
# SHARED UTILITY: APPLY FRICTION
# ==========================================
def apply_entry_friction(price):
    return price * (1 + SLIPPAGE_ENTRY)

def apply_exit_friction(price):
    return price * (1 - SLIPPAGE_EXIT)

# ==========================================
# SHARED UTILITY: CASH DRAG
# ==========================================
def apply_cash_drag(cash):
    return cash * (1 + DAILY_RF)

# ==========================================
# SHARED UTILITY: GAP-DOWN EXIT
# ==========================================
def gap_down_exit(open_price, stop_price, close_price):
    """Exit fill model for the day a position is closed.
    FIXED: previously every call site passed open_price as BOTH the open
    and the close argument, so this function always returned open_price no
    matter which branch ran -- the close-price branch was dead code and
    every exit silently filled at the day's open regardless of what
    actually happened intraday.
    Now: if the day gaps below the stop at the open, you're filled at that
    (worse) gapped-down open price, same as a real stop order would be.
    Otherwise, the position is assumed held through the day and exited at
    that day's close (matches the live script's MOC -- market-on-close --
    exit routing)."""
    if open_price < stop_price:
        return open_price
    return close_price

# ==========================================
# POSITION SIZING
# ==========================================
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

# ==========================================
# SECTOR CAP CHECK
# ==========================================
def sector_cap_reached(open_positions, sector):
    count = sum(1 for p in open_positions.values() if p['sector'] == sector)
    return count >= MAX_SECTOR_POSITIONS

# ==========================================
# MODEL A — YANG (Trend Following) - RANKED
# ==========================================
def run_yang(ind, data, sp500, sp500_sma200, sp500_sma50):
    cash = START_CAPITAL
    equity = START_CAPITAL
    open_positions = {}
    trade_log = []
    equity_curve = []
    daily_returns = []

    dates = ind[tickers[0]].index[200:]

    for i in range(1, len(dates)):
        date = dates[i]
        prev_date = dates[i-1]

        cash = apply_cash_drag(cash)

        market_ok = (sp500.loc[prev_date] > sp500_sma200.loc[prev_date]) and \
                    (sp500.loc[prev_date] > sp500_sma50.loc[prev_date])

        # Exits
        to_close = []
        for t, pos in open_positions.items():
            df = ind[t]
            prev_close = df.loc[prev_date, 'Close']
            prev_sma50 = df.loc[prev_date, 'SMA50']

            if prev_close < prev_sma50 or prev_close <= pos['stop']:
                to_close.append(t)

        for t in to_close:
            pos = open_positions.pop(t)
            open_price = data['Open'][t].loc[date]
            close_price = ind[t].loc[date, 'Close']
            exit_price = gap_down_exit(open_price, pos['stop'], close_price)
            exit_price = apply_exit_friction(exit_price)

            cash += pos['shares'] * exit_price
            cash -= COMMISSION

            pnl = (exit_price - pos['entry']) * pos['shares'] - COMMISSION
            trade_log.append({
                'ticker': t,
                'entry_date': pos['entry_date'],
                'exit_date': date,
                'entry_price': pos['entry'],
                'exit_price': exit_price,
                'shares': pos['shares'],
                'pnl': pnl
            })

        # Entries (ranked by strength)
        if market_ok:
            available_slots = MAX_TOTAL_POSITIONS - len(open_positions)
            candidates = []

            for t in tickers:
                if t in open_positions:
                    continue

                df = ind[t]
                y_close = df.loc[prev_date, 'Close']
                y_sma50 = df.loc[prev_date, 'SMA50']
                y_prior_high = df.loc[prev_date, 'Prior20High']
                y_vol = df.loc[prev_date, 'Vol']
                y_avgvol30 = df.loc[prev_date, 'AvgVol30']
                y_atr = df.loc[prev_date, 'ATR14']
                y_sma20 = df.loc[prev_date, 'SMA20']

                sector = sectors[t]
                if sector_cap_reached(open_positions, sector):
                    continue

                trigger_entry = False
                mod_type = ""
                stop_mult = 2.0
                strength = 0.0

                # High Vol Breakout
                if y_close > y_sma50 and y_vol > y_avgvol30 and y_close > y_prior_high:
                    trigger_entry = True
                    mod_type = "Yang Module A (High Vol Breakout)"
                    strength = (y_close - y_prior_high) / y_atr if y_atr > 0 else 0
                # Low Vol Pullback
                elif y_close > y_sma50 and y_vol < y_avgvol30 and y_close <= (y_sma20 + 2 * y_atr):
                    trigger_entry = True
                    mod_type = "Yang Module B (Low Vol Pullback)"
                    strength = (y_sma50 - y_close) / y_atr if y_atr > 0 else 0

                if trigger_entry:
                    candidates.append({
                        'ticker': t,
                        'mod_type': mod_type,
                        'stop_mult': stop_mult,
                        'atr': y_atr,
                        'strength': strength,
                        'sector': sector
                    })

            candidates.sort(key=lambda x: x['strength'], reverse=True)

            for c in candidates:
                if available_slots <= 0:
                    break

                t = c['ticker']
                sector = c['sector']
                if sector_cap_reached(open_positions, sector):
                    continue

                raw_open = data['Open'][t].loc[date]
                stop = raw_open - (c['stop_mult'] * c['atr'])
                shares = position_size(raw_open, stop, equity)

                entry_price = apply_entry_friction(raw_open)
                total_cost = (shares * entry_price) + COMMISSION

                if shares > 0 and cash >= total_cost:
                    cash -= total_cost
                    open_positions[t] = {
                        'entry': entry_price,
                        'stop': stop,
                        'shares': shares,
                        'sector': sector,
                        'entry_date': date
                    }
                    available_slots -= 1

        open_val = sum(
            open_positions[t]['shares'] * ind[t].loc[date, 'Close']
            for t in open_positions
        )
        prev_equity = equity
        equity = cash + open_val

        if i > 1:
            daily_returns.append((equity - prev_equity) / prev_equity)

        equity_curve.append(equity)

    return {
        'name': 'Yang',
        'equity_curve': equity_curve,
        'trade_log': trade_log,
        'daily_returns': daily_returns,
        'dates': list(dates[1:])
    }

# ==========================================
# MODEL C — YIN (Mean Reversion) - VOL BREAKER + RANKED
# ==========================================
def run_yin(ind, data, sp500, sp500_sma200, sp500_sma50):
    cash = START_CAPITAL
    equity = START_CAPITAL
    open_positions = {}
    trade_log = []
    equity_curve = []
    daily_returns = []

    dates = ind[tickers[0]].index[200:]

    for i in range(1, len(dates)):
        date = dates[i]
        prev_date = dates[i-1]

        cash = apply_cash_drag(cash)

        market_ok = (sp500.loc[prev_date] > sp500_sma200.loc[prev_date]) and \
                    (sp500.loc[prev_date] > sp500_sma50.loc[prev_date])

        # Exits + time-stop
        to_close = []
        for t, pos in open_positions.items():
            df = ind[t]
            prev_close = df.loc[prev_date, 'Close']
            prev_sma5 = df.loc[prev_date, 'SMA5']

            if pos['days_held'] >= 4:
                to_close.append(t)
            elif prev_close > prev_sma5 or prev_close <= pos['stop']:
                to_close.append(t)

        for t in to_close:
            pos = open_positions.pop(t)
            open_price = data['Open'][t].loc[date]
            close_price = ind[t].loc[date, 'Close']
            exit_price = gap_down_exit(open_price, pos['stop'], close_price)
            exit_price = apply_exit_friction(exit_price)

            cash += pos['shares'] * exit_price
            cash -= COMMISSION

            pnl = (exit_price - pos['entry']) * pos['shares'] - COMMISSION
            trade_log.append({
                'ticker': t,
                'entry_date': pos['entry_date'],
                'exit_date': date,
                'entry_price': pos['entry'],
                'exit_price': exit_price,
                'shares': pos['shares'],
                'pnl': pnl
            })

        # Entries (volatility breaker + ranked)
        if market_ok:
            available_slots = MAX_TOTAL_POSITIONS - len(open_positions)
            candidates = []

            for t in tickers:
                if available_slots <= 0:
                    break
                if t in open_positions:
                    continue

                df = ind[t]
                y_close = df.loc[prev_date, 'Close']
                y_rsi2 = df.loc[prev_date, 'RSI2']
                y_sma200 = df.loc[prev_date, 'SMA200']
                y_atr14 = df.loc[prev_date, 'ATR14']
                y_atr50 = df.loc[prev_date, 'ATR50']

                sector = sectors[t]
                if sector_cap_reached(open_positions, sector):
                    continue

                # Volatility breaker: skip if ATR14 exploding vs ATR50
                if pd.notna(y_atr50) and y_atr50 > 0 and y_atr14 > 2 * y_atr50:
                    continue

                trigger_entry = False
                mod_type = ""
                strength = 0.0

                if y_close > y_sma200 and y_rsi2 < 5:
                    trigger_entry = True
                    mod_type = "Yin Module B (Deep RSI < 5)"
                    strength = 15 - y_rsi2
                elif y_close > y_sma200 and y_rsi2 < 10:
                    trigger_entry = True
                    mod_type = "Yin Module A (Standard RSI < 10)"
                    strength = 10 - y_rsi2

                if trigger_entry:
                    candidates.append({
                        'ticker': t,
                        'mod_type': mod_type,
                        'atr': y_atr14,
                        'strength': strength,
                        'sector': sector
                    })

            candidates.sort(key=lambda x: x['strength'], reverse=True)

            for c in candidates:
                if available_slots <= 0:
                    break

                t = c['ticker']
                sector = c['sector']
                if sector_cap_reached(open_positions, sector):
                    continue

                raw_open = data['Open'][t].loc[date]
                stop = raw_open - (1.0 * c['atr'])
                shares = position_size(raw_open, stop, equity)

                entry_price = apply_entry_friction(raw_open)
                total_cost = (shares * entry_price) + COMMISSION

                if shares > 0 and cash >= total_cost:
                    cash -= total_cost
                    open_positions[t] = {
                        'entry': entry_price,
                        'stop': stop,
                        'shares': shares,
                        'sector': sector,
                        'entry_date': date,
                        'days_held': 0
                    }
                    available_slots -= 1

        for t in open_positions:
            open_positions[t]['days_held'] += 1

        open_val = sum(
            open_positions[t]['shares'] * ind[t].loc[date, 'Close']
            for t in open_positions
        )
        prev_equity = equity
        equity = cash + open_val

        if i > 1:
            daily_returns.append((equity - prev_equity) / prev_equity)

        equity_curve.append(equity)

    return {
        'name': 'Yin',
        'equity_curve': equity_curve,
        'trade_log': trade_log,
        'daily_returns': daily_returns,
        'dates': list(dates[1:])
    }

# ==========================================
# MODEL B — ZHAN (Volatility Expansion) - RANKED
# ==========================================
def run_zhan(ind, data, sp500, sp500_sma200, sp500_sma50):
    cash = START_CAPITAL
    equity = START_CAPITAL
    open_positions = {}
    trade_log = []
    equity_curve = []
    daily_returns = []

    dates = ind[tickers[0]].index[200:]

    for i in range(1, len(dates)):
        date = dates[i]
        prev_date = dates[i-1]

        cash = apply_cash_drag(cash)

        market_ok = (sp500.loc[prev_date] > sp500_sma200.loc[prev_date]) and \
                    (sp500.loc[prev_date] > sp500_sma50.loc[prev_date])

        # Exits
        to_close = []
        for t, pos in open_positions.items():
            df = ind[t]
            prev_close = df.loc[prev_date, 'Close']
            prev_kc_lower = df.loc[prev_date, 'KC_LOWER']

            if prev_close < prev_kc_lower or prev_close <= pos['stop']:
                to_close.append(t)

        for t in to_close:
            pos = open_positions.pop(t)
            open_price = data['Open'][t].loc[date]
            close_price = ind[t].loc[date, 'Close']
            exit_price = gap_down_exit(open_price, pos['stop'], close_price)
            exit_price = apply_exit_friction(exit_price)

            cash += pos['shares'] * exit_price
            cash -= COMMISSION

            pnl = (exit_price - pos['entry']) * pos['shares'] - COMMISSION
            trade_log.append({
                'ticker': t,
                'entry_date': pos['entry_date'],
                'exit_date': date,
                'entry_price': pos['entry'],
                'exit_price': exit_price,
                'shares': pos['shares'],
                'pnl': pnl
            })

        # Entries (ranked)
        if market_ok:
            available_slots = MAX_TOTAL_POSITIONS - len(open_positions)
            candidates = []

            for t in tickers:
                if t in open_positions:
                    continue

                df = ind[t]
                curr_idx = df.index.get_loc(prev_date)

                y_close = df.loc[prev_date, 'Close']
                y_ema20 = df.loc[prev_date, 'EMA20']
                y_kc_upper = df.loc[prev_date, 'KC_UPPER']
                y_vol = df.loc[prev_date, 'Vol']
                y_avgvol30 = df.loc[prev_date, 'AvgVol30']
                y_atr = df.loc[prev_date, 'ATR14']

                sector = sectors[t]
                if sector_cap_reached(open_positions, sector):
                    continue

                cons = (
                    df.iloc[curr_idx]['Close'] < df.iloc[curr_idx]['KC_UPPER'] and
                    df.iloc[curr_idx]['Close'] > df.iloc[curr_idx]['KC_LOWER'] and
                    df.iloc[curr_idx-1]['Close'] < df.iloc[curr_idx-1]['KC_UPPER'] and
                    df.iloc[curr_idx-1]['Close'] > df.iloc[curr_idx-1]['KC_LOWER'] and
                    df.iloc[curr_idx-2]['Close'] < df.iloc[curr_idx-2]['KC_UPPER'] and
                    df.iloc[curr_idx-2]['Close'] > df.iloc[curr_idx-2]['KC_LOWER']
                )

                trigger_entry = False
                mod_type = ""
                strength = 0.0

                # High Vol Keltner Breakout
                if y_close > y_kc_upper and y_vol > y_avgvol30 and cons:
                    trigger_entry = True
                    mod_type = "Zhan Module A (Keltner Breakout)"
                    strength = (y_close - y_kc_upper) / y_atr if y_atr > 0 else 0
                # Low Vol EMA Pullback
                elif y_close > y_ema20 and y_vol < y_avgvol30 and cons:
                    trigger_entry = True
                    mod_type = "Zhan Module B (EMA Breakout)"
                    strength = (y_close - y_ema20) / y_atr if y_atr > 0 else 0

                if trigger_entry:
                    candidates.append({
                        'ticker': t,
                        'mod_type': mod_type,
                        'atr': y_atr,
                        'strength': strength,
                        'sector': sector
                    })

            candidates.sort(key=lambda x: x['strength'], reverse=True)

            for c in candidates:
                if available_slots <= 0:
                    break

                t = c['ticker']
                sector = c['sector']
                if sector_cap_reached(open_positions, sector):
                    continue

                raw_open = data['Open'][t].loc[date]
                stop = raw_open - (1.5 * c['atr'])
                shares = position_size(raw_open, stop, equity)

                entry_price = apply_entry_friction(raw_open)
                total_cost = (shares * entry_price) + COMMISSION

                if shares > 0 and cash >= total_cost:
                    cash -= total_cost
                    open_positions[t] = {
                        'entry': entry_price,
                        'stop': stop,
                        'shares': shares,
                        'sector': sector,
                        'entry_date': date
                    }
                    available_slots -= 1

        open_val = sum(
            open_positions[t]['shares'] * ind[t].loc[date, 'Close']
            for t in open_positions
        )
        prev_equity = equity
        equity = cash + open_val

        if i > 1:
            daily_returns.append((equity - prev_equity) / prev_equity)

        equity_curve.append(equity)

    return {
        'name': 'Zhan',
        'equity_curve': equity_curve,
        'trade_log': trade_log,
        'daily_returns': daily_returns,
        'dates': list(dates[1:])
    }

# ==========================================
# BENCHMARK — SPY / S&P 500 BUY & HOLD
# ==========================================
def run_benchmark(sp500, dates):
    """Buy-and-hold the index over the same dates the strategies traded,
    starting from the same capital. No trade log, no friction/commission
    (that's standard for a passive benchmark comparison)."""
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
        'name': 'SPX Buy & Hold',
        'equity_curve': equity_curve,
        'trade_log': [],
        'daily_returns': daily_returns,
        'dates': list(dates[1:])
    }

# ==========================================
# PERFORMANCE METRICS ENGINE
# ==========================================
def compute_metrics(model):
    eq = pd.Series(model['equity_curve'])
    daily = np.array(model['daily_returns'])
    trades = model['trade_log']

    running_peak = eq.cummax()
    drawdown = (eq - running_peak) / running_peak
    max_dd = drawdown.min() * 100

    total_return = (eq.iloc[-1] - START_CAPITAL) / START_CAPITAL * 100

    daily_std = np.std(daily)
    if daily_std != 0:
        sharpe = (np.mean(daily) / daily_std) * np.sqrt(252)
    else:
        sharpe = 0

    downside_returns = np.where(daily < 0, daily, 0)
    downside_dev = np.std(downside_returns)

    if downside_dev > 0:
        sortino = (np.mean(daily) / downside_dev) * np.sqrt(252)
    else:
        sortino = 0

    gross_profit = sum(tr['pnl'] for tr in trades if tr['pnl'] > 0)
    gross_loss = -sum(tr['pnl'] for tr in trades if tr['pnl'] < 0)
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.nan

    wins = sum(1 for tr in trades if tr['pnl'] > 0)
    total_trades = len(trades)
    win_rate = (wins / total_trades * 100) if total_trades > 0 else np.nan

    return {
        'name': model['name'],
        'total_return': total_return,
        'max_dd': max_dd,
        'sharpe': sharpe,
        'sortino': sortino,
        'profit_factor': profit_factor,
        'win_rate': win_rate,
        'total_trades': total_trades,
        'equity_curve': eq,
        'drawdown': drawdown,
        'trade_log': trades
    }

# ==========================================
# MONTE CARLO — BLOCK BOOTSTRAP ON DAILY RETURNS
# ==========================================
def monte_carlo_bootstrap(daily_returns, n_sims=5000, block_size=5, seed=42):
    """Resample the strategy's own historical daily returns in contiguous
    blocks (to preserve autocorrelation/streaks from multi-day holds) to
    build n_sims synthetic equity paths of the same length as the original.

    This is a robustness/confidence-interval check on the sample you
    already have -- it tells you how much of your single historical result
    could plausibly have come out differently just from the order days
    happened to fall in. It does NOT simulate unseen future market
    conditions or new price data.
    """
    rng = np.random.default_rng(seed)
    daily = np.asarray(daily_returns, dtype=float)
    n = len(daily)
    if n == 0:
        return None

    results = {'total_return': [], 'max_dd': [], 'sharpe': []}

    for _ in range(n_sims):
        path = []
        while len(path) < n:
            start = rng.integers(0, n - block_size + 1) if n > block_size else 0
            block = daily[start:start + block_size]
            path.extend(block)
        path = np.array(path[:n])

        eq = START_CAPITAL * np.cumprod(1 + path)
        running_peak = np.maximum.accumulate(eq)
        drawdown = (eq - running_peak) / running_peak
        max_dd = drawdown.min() * 100

        total_return = (eq[-1] - START_CAPITAL) / START_CAPITAL * 100

        std = np.std(path)
        sharpe = (np.mean(path) / std) * np.sqrt(252) if std != 0 else 0

        results['total_return'].append(total_return)
        results['max_dd'].append(max_dd)
        results['sharpe'].append(sharpe)

    return {k: np.array(v) for k, v in results.items()}

def summarize_monte_carlo(name, mc_results, actual_total_return):
    tr = mc_results['total_return']
    dd = mc_results['max_dd']
    sh = mc_results['sharpe']

    pct_positive = (tr > 0).mean() * 100
    actual_percentile = (tr < actual_total_return).mean() * 100

    return {
        'name': name,
        'return_p5': np.percentile(tr, 5),
        'return_p25': np.percentile(tr, 25),
        'return_median': np.percentile(tr, 50),
        'return_p75': np.percentile(tr, 75),
        'return_p95': np.percentile(tr, 95),
        'dd_p5': np.percentile(dd, 5),      # worst-case tail (5th pct of a negative number)
        'dd_median': np.percentile(dd, 50),
        'sharpe_median': np.percentile(sh, 50),
        'pct_sims_positive': pct_positive,
        'actual_return': actual_total_return,
        'actual_percentile_in_dist': actual_percentile,
    }

def run_monte_carlo_suite(results, n_sims=5000, block_size=5, seed=42):
    """Runs bootstrap MC for every model in `results` (including the
    benchmark) and prints a summary table plus a pairwise win-rate of each
    strategy vs the benchmark, drawn from independently resampled paths."""
    summaries = []
    mc_by_name = {}

    for r in results:
        actual_total_return = (pd.Series(r['equity_curve']).iloc[-1] - START_CAPITAL) / START_CAPITAL * 100
        mc = monte_carlo_bootstrap(r['daily_returns'], n_sims=n_sims, block_size=block_size, seed=seed)
        if mc is None:
            continue
        mc_by_name[r['name']] = mc
        summaries.append(summarize_monte_carlo(r['name'], mc, actual_total_return))

    table = pd.DataFrame(summaries)
    print("\n==============================")
    print(f"MONTE CARLO (block bootstrap, {n_sims} sims, block size {block_size} days)")
    print("==============================")
    print_cols = ['name', 'return_p5', 'return_p25', 'return_median', 'return_p75',
                  'return_p95', 'dd_p5', 'sharpe_median', 'pct_sims_positive', 'actual_return']
    rename = {
        'name': 'Model', 'return_p5': 'Return P5%', 'return_p25': 'Return P25%',
        'return_median': 'Return Median%', 'return_p75': 'Return P75%',
        'return_p95': 'Return P95%', 'dd_p5': 'Worst-Case DD% (P5)',
        'sharpe_median': 'Median Sharpe', 'pct_sims_positive': '% Sims Profitable',
        'actual_return': 'Actual Return%'
    }
    print(table[print_cols].rename(columns=rename).round(2).to_string(index=False))

    # Pairwise: what fraction of independently-resampled strategy paths beat
    # an independently-resampled SPX path?
    if 'SPX Buy & Hold' in mc_by_name:
        bench_tr = mc_by_name['SPX Buy & Hold']['total_return']
        print("\n==============================")
        print("P(STRATEGY PATH BEATS SPX PATH) — from independently resampled Monte Carlo runs")
        print("==============================")
        for name, mc in mc_by_name.items():
            if name == 'SPX Buy & Hold':
                continue
            n = min(len(mc['total_return']), len(bench_tr))
            win_rate = (mc['total_return'][:n] > bench_tr[:n]).mean() * 100
            print(f"{name:>6}: beats SPX in {win_rate:.1f}% of paired simulations")

    return table


def compare_models(results, SHOW_CHARTS=False):
    metrics = [compute_metrics(r) for r in results]

    if SHOW_CHARTS:
        plt.figure(figsize=(12,6))
        for m in metrics:
            plt.plot(m['equity_curve'], label=m['name'])
        plt.title("Equity Curves — Yang vs Zhan vs Yin vs SPX Buy & Hold")
        plt.legend()
        plt.grid(True)
        plt.show()

        plt.figure(figsize=(12,4))
        for m in metrics:
            plt.plot(m['drawdown'] * 100, label=m['name'])
        plt.title("Drawdown Comparison (%)")
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

    print("\n==============================")
    print("MODEL PERFORMANCE COMPARISON (incl. SPX benchmark)")
    print("==============================")
    print(table.to_string(index=False))

    print("\n==============================")
    print("STRATEGY RANKING (by Total Return)")
    print("==============================")
    ranked = table.sort_values(by='Total Return %', ascending=False)
    print(ranked[['Model', 'Total Return %']].to_string(index=False))

    # Alpha vs. benchmark, if present
    bench_row = table[table['Model'] == 'SPX Buy & Hold']
    if not bench_row.empty:
        bench_return = bench_row['Total Return %'].values[0]
        bench_sharpe = bench_row['Sharpe'].values[0]
        print("\n==============================")
        print("ALPHA vs SPX BUY & HOLD")
        print("==============================")
        for m in metrics:
            if m['name'] == 'SPX Buy & Hold':
                continue
            excess_return = m['total_return'] - bench_return
            excess_sharpe = m['sharpe'] - bench_sharpe
            print(f"{m['name']:>6}: {excess_return:+7.2f}% total return vs SPX | "
                  f"{excess_sharpe:+.2f} Sharpe vs SPX")

    for m in metrics:
        if not m['trade_log']:
            continue  # benchmark has no trades to export
        df = pd.DataFrame(m['trade_log'])
        df.to_csv(f"{m['name'].lower().replace(' ', '_')}_trades.csv", index=False)
        print(f"Exported {m['name']} trades → {m['name'].lower().replace(' ', '_')}_trades.csv")

# ==========================================
# MAIN EXECUTION PIPELINE
# ==========================================
def run_full_suite(ind, data, sp500, sp500_sma200, sp500_sma50,
                    max_total_positions, max_alloc_fraction, max_sector_positions,
                    label=""):
    """Runs Yang/Zhan/Yin with the given position-limit parameters. These
    three parameters (not RISK_PER_TRADE, which is a real risk-sizing rule)
    were originally set low to make daily manual data collection/tracking
    feasible -- they're not necessarily reflective of the strategy logic
    itself. Mutates the module-level globals for the duration of the run so
    run_yang/run_zhan/run_yin (which read them as globals) pick up the
    values, then restores them."""
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

if __name__ == "__main__":
    data = load_data()
    ind, sp500, sp500_sma200, sp500_sma50 = build_indicators(data)
    dates = ind[tickers[0]].index[200:]

    print("\n### RUN 1: Manual-Tracking Limits (original) ###")
    print(f"    MAX_TOTAL_POSITIONS=6, MAX_ALLOC_FRACTION=1/6, MAX_SECTOR_POSITIONS=2")
    yang_manual, zhan_manual, yin_manual = run_full_suite(
        ind, data, sp500, sp500_sma200, sp500_sma50,
        max_total_positions=6, max_alloc_fraction=1/6, max_sector_positions=2,
        label="(Manual Limits)"
    )

    print("\n### RUN 2: Unconstrained (no manual-tracking caps) ###")
    print(f"    MAX_TOTAL_POSITIONS={len(tickers)} (one slot per ticker), "
          f"MAX_ALLOC_FRACTION=1.0 (position size driven purely by 1% risk-per-trade sizing), "
          f"MAX_SECTOR_POSITIONS={len(tickers)} (no sector cap)")
    yang_free, zhan_free, yin_free = run_full_suite(
        ind, data, sp500, sp500_sma200, sp500_sma50,
        max_total_positions=len(tickers), max_alloc_fraction=1.0, max_sector_positions=len(tickers),
        label="(Unconstrained)"
    )

    print("\nProcessing SPX Buy & Hold benchmark...")
    benchmark_results = run_benchmark(sp500, dates)

    all_results = [
        yang_manual, zhan_manual, yin_manual,
        yang_free, zhan_free, yin_free,
        benchmark_results
    ]

    compare_models(all_results, SHOW_CHARTS=False)

    print("\nRunning Monte Carlo block-bootstrap analysis...")
    run_monte_carlo_suite(all_results, n_sims=5000, block_size=5, seed=42)
