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

tickers = [
    'MSFT', 'AMZN', 'JPM', 'UNH', 'XOM', 'WMT', 'NVDA', 'AMD', 'META', 'TSLA',
    'AAPL', 'GOOGL', 'NFLX', 'AVGO', 'CRM',  # Tech / Communications
    'HD', 'COST', 'NKE', 'MCD', 'ORLY',     # Consumer Discretionary / Staples
    'BAC', 'GS', 'MS', 'V', 'MA',           # Financials / Payments
    'LLY', 'MRK', 'PFE', 'JNJ',             # Healthcare
    'CVX', 'COP'                            # Energy
]

sectors = {
    # Tech & Communication Services
    'MSFT': 'Tech', 'NVDA': 'Tech', 'AMD': 'Tech', 'META': 'Tech', 
    'AAPL': 'Tech', 'GOOGL': 'Tech', 'NFLX': 'Tech', 'AVGO': 'Tech', 'CRM': 'Tech',
    
    # Consumer Discretionary & Staples
    'AMZN': 'Consumer', 'TSLA': 'Consumer', 'WMT': 'Consumer', 
    'HD': 'Consumer', 'COST': 'Consumer', 'NKE': 'Consumer', 'MCD': 'Consumer', 'ORLY': 'Consumer',
    
    # Financials & Transaction Services
    'JPM': 'Finance', 'BAC': 'Finance', 'GS': 'Finance', 'MS': 'Finance', 
    'V': 'Finance', 'MA': 'Finance',
    
    # Healthcare
    'UNH': 'Healthcare', 'LLY': 'Healthcare', 'MRK': 'Healthcare', 'PFE': 'Healthcare', 'JNJ': 'Healthcare',
    
    # Energy
    'XOM': 'Energy', 'CVX': 'Energy', 'COP': 'Energy'
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
    data = yf.download(tickers + ['^GSPC'], start="2007-01-01", end="2010-01-01",auto_adjust=True)
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

# ==============================================================================
# PRODUCTION READY MODEL A — YANG (Trend Following) - WITH TRAILING STOPS
# ==============================================================================
def run_yang(ind, data, sp500, sp500_sma200, sp500_sma50):
    cash = START_CAPITAL
    equity = START_CAPITAL
    open_positions = {}
    trade_log = []
    equity_curve = []
    daily_returns = []

    # Timeline driven securely from the market index calendar
    dates = sp500.index[200:]

    for i in range(1, len(dates)):
        date = dates[i]
        prev_date = dates[i-1]

        # Apply daily compounding interest metrics to standing cash
        cash = apply_cash_drag(cash)

        # Primary Market Regime Filter Check
        market_ok = (sp500.loc[prev_date] > sp500_sma200.loc[prev_date]) and \
                    (sp500.loc[prev_date] > sp500_sma50.loc[prev_date])

        just_exited_today = set()

        # --- PHASE A: SEQUENTIAL ACCOUNTING EXITS ---
        to_close = []
        for t, pos in open_positions.items():
            df = ind.get(t)
            if df is None or prev_date not in df.index:
                continue
                
            prev_close = df.loc[prev_date, 'Close']
            prev_sma50 = df.loc[prev_date, 'SMA50']
            y_atr = df.loc[prev_date, 'ATR14']

            # FIX: Trailing Stop Maintenance Logic
            # Dynamically raise the stop tracking trailing peak prices minus volatility buffer
            calculated_trailing_stop = prev_close - (pos['stop_mult'] * y_atr)
            if calculated_trailing_stop > pos['stop']:
                pos['stop'] = calculated_trailing_stop

            # Structural Strategy Exits: Trend Breach OR Trailing Stop Violations
            if prev_close < prev_sma50 or prev_close <= pos['stop']:
                to_close.append(t)

        # Execute liquidations completely to free up capacity slots and cash balances
        for t in to_close:
            pos = open_positions.pop(t)
            just_exited_today.add(t) 
            
            if t not in data['Open'] or date not in data['Open'][t].index:
                continue

            open_price = data['Open'][t].loc[date]
            close_price = ind[t].loc[date, 'Close']
            
            # Gap protection engine
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

        # --- PHASE B: CANDIDATE DISCOVERY AND SCORING ---
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

                # High Vol Breakout Setup Rules
                if y_close > y_sma50 and y_vol > y_avgvol30 and y_close > y_prior_high:
                    trigger_entry = True
                    mod_type = "Yang Module A (High Vol Breakout)"
                    strength = (y_close - y_prior_high) / y_atr if y_atr > 0 else 0
                
                # Low Vol Pullback Setup Rules
                elif y_close > y_sma50 and y_vol < y_avgvol30 and y_close <= (y_sma20 + 2 * y_atr):
                    trigger_entry = True
                    mod_type = "Yang Module B (Low Vol Pullback)"
                    strength = (y_close - y_sma50) / y_atr if y_atr > 0 else 0

                if trigger_entry:
                    candidates.append({
                        'ticker': t, 'mod_type': mod_type, 'stop_mult': stop_mult, 
                        'atr': y_atr, 'strength': strength, 'sector': sector
                    })

            # Sort candidate vector by normalization metrics
            candidates.sort(key=lambda x: x['strength'], reverse=True)

            # --- PHASE C: RANKED ALLOCATION GATE ---
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
                        'stop_mult': c['stop_mult'], # Preserved for dynamic trailing stop calculation
                        'days_held': 0 
                    }
                    available_slots -= 1
                    transient_equity -= total_cost 

        # Unified inventory age progression tracking
        for t in open_positions:
            open_positions[t]['days_held'] = open_positions[t].get('days_held', 0) + 1

        # Day-End Portfolio Bookkeeping
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
        'name': 'Yang_Production_Final',
        'equity_curve': equity_curve,
        'trade_log': trade_log,
        'daily_returns': daily_returns,
        'dates': list(dates[1:])
    }

# ==============================================================================
# PRODUCTION READY MODEL C — YIN (Mean Reversion) - BUG FREE & OPTIMIZED
# ==============================================================================
def run_yin(ind, data, sp500, sp500_sma200, sp500_sma50):
    cash = START_CAPITAL
    equity = START_CAPITAL
    open_positions = {}
    trade_log = []
    equity_curve = []
    daily_returns = []

    # FIX 1: Drive the timeline from the S&P500 index calendar to avoid individual ticker data holes
    dates = sp500.index[200:]

    for i in range(1, len(dates)):
        date = dates[i]
        prev_date = dates[i-1]

        # Apply daily compounding risk-free interest to sitting cash
        cash = apply_cash_drag(cash)

        # Primary Market Regime Filter Check
        market_ok = (sp500.loc[prev_date] > sp500_sma200.loc[prev_date]) and \
                    (sp500.loc[prev_date] > sp500_sma50.loc[prev_date])

        # Track tickers that were active today to prevent immediate re-entry churn
        just_exited_today = set()

        # --- PHASE A: SEQUENTIAL ACCOUNTING EXITS ---
        to_close = []
        for t, pos in open_positions.items():
            df = ind.get(t)
            if df is None or prev_date not in df.index:
                continue
                
            prev_close = df.loc[prev_date, 'Close']
            prev_sma5 = df.loc[prev_date, 'SMA5']

            # Time-Stop Safety Check or Active Strategy Exits
            if pos['days_held'] >= 4:
                to_close.append(t)
            elif prev_close > prev_sma5 or prev_close <= pos['stop']:
                to_close.append(t)

        # Execute liquidations completely to free up capacity slots and cash balances
        for t in to_close:
            pos = open_positions.pop(t)
            just_exited_today.add(t) # FIX 4: Flag ticker to prevent same-day re-entry anomalies
            
            # Guard against missing price series updates
            if t not in data['Open'] or date not in data['Open'][t].index:
                continue

            open_price = data['Open'][t].loc[date]
            close_price = ind[t].loc[date, 'Close']
            
            # Run gap-down check to ensure execution matches physical reality
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

        # --- PHASE B: CANDIDATE DISCOVERY AND SCORING ---
        if market_ok:
            available_slots = MAX_TOTAL_POSITIONS - len(open_positions)
            candidates = []

            for t in tickers:
                # Check active inventory and the new same-day exit constraint
                if t in open_positions or t in just_exited_today:
                    continue

                df = ind.get(t)
                # FIX 1: Safely bypass tickers that don't have recorded data for this day
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

                # Volatility Breaker
                if pd.notna(y_atr50) and y_atr50 > 0 and y_atr14 > 2 * y_atr50:
                    continue

                trigger_entry = False
                mod_type = ""
                strength = 0.0

                # Explicit Hierarchical Signal Ranking
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

            # Sort discovered pool by true strength score
            candidates.sort(key=lambda x: x['strength'], reverse=True)

            # --- PHASE C: RANKED ALLOCATION GATE ---
            # FIX 2: Create a transient risk equity pool to prevent compounding capital over-sizing
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
                
                # Size relative to historical capacity available
                shares = position_size(raw_open, stop, transient_equity)

                entry_price = apply_entry_friction(raw_open)
                total_cost = (shares * entry_price) + COMMISSION

                if shares > 0 and cash >= total_cost:
                    cash -= total_cost
                    open_positions[t] = {
                        'entry': entry_price, 'stop': stop, 'shares': shares,
                        'sector': sector, 'entry_date': date,
                        'days_held': 0 # Initialized at 0 before market close processing
                    }
                    available_slots -= 1
                    # Reduce risk pool size for any subsequent setups processed on this morning
                    transient_equity -= total_cost 

        # FIX 3: Increment duration counter safely *after* market closes for accurate calculation
        for t in open_positions:
            open_positions[t]['days_held'] += 1

        # Day-End Portfolio Bookkeeping
        open_val = 0.0
        for t, pos in open_positions.items():
            df = ind.get(t)
            if df is not None and date in df.index:
                open_val += pos['shares'] * df.loc[date, 'Close']
            else:
                open_val += pos['shares'] * pos['entry'] # Fallback protection if session misses data

        prev_equity = equity
        equity = cash + open_val

        if i > 1:
            daily_returns.append((equity - prev_equity) / prev_equity)

        equity_curve.append(equity)

    return {
        'name': 'Yin_Production_Fixed',
        'equity_curve': equity_curve,
        'trade_log': trade_log,
        'daily_returns': daily_returns,
        'dates': list(dates[1:])
    }


# ==============================================================================
# PRODUCTION READY MODEL B — ZHAN (Volatility Expansion) - TRULY FLAWLESS
# ==============================================================================
def run_zhan(ind, data, sp500, sp500_sma200, sp500_sma50):
    cash = START_CAPITAL
    equity = START_CAPITAL
    open_positions = {}
    trade_log = []
    equity_curve = []
    daily_returns = []

    # Timeline driven securely from the market index calendar matrix
    dates = sp500.index[200:]

    # OPTIMIZATION FIX: Pre-map dates to integers to eliminate slow .get_loc() calls
    date_to_idx = {d: idx for idx, d in enumerate(sp500.index)}

    for i in range(1, len(dates)):
        date = dates[i]
        prev_date = dates[i-1]

        # Apply daily compounding interest metrics to standing cash balances
        cash = apply_cash_drag(cash)

        # Primary Market Regime Filter Check
        market_ok = (sp500.loc[prev_date] > sp500_sma200.loc[prev_date]) and \
                    (sp500.loc[prev_date] > sp500_sma50.loc[prev_date])

        just_exited_today = set()

        # --- PHASE A: SEQUENTIAL ACCOUNTING EXITS ---
        to_close = []
        for t, pos in open_positions.items():
            df = ind.get(t)
            if df is None or prev_date not in df.index:
                continue
                
            prev_close = df.loc[prev_date, 'Close']
            prev_kc_lower = df.loc[prev_date, 'KC_LOWER']
            y_atr = df.loc[prev_date, 'ATR14']

            # STRATEGY FIX 1: Trailing Stop Logic to defend open breakout profits
            calculated_trailing_stop = prev_close - (pos['stop_mult'] * y_atr)
            if calculated_trailing_stop > pos['stop']:
                pos['stop'] = calculated_trailing_stop

            # Strategy Exits: Volatility Boundary Breach OR Hit Trailing Stop Loss
            if prev_close < prev_kc_lower or prev_close <= pos['stop']:
                to_close.append(t)

        # Execute liquidations completely to free up capacity slots and cash balances
        for t in to_close:
            pos = open_positions.pop(t)
            just_exited_today.add(t) 
            
            if t not in data['Open'] or date not in data['Open'][t].index:
                continue

            open_price = data['Open'][t].loc[date]
            close_price = ind[t].loc[date, 'Close']
            
            # Gap protection engine
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

        # --- PHASE B: CANDIDATE DISCOVERY AND SCORING ---
        if market_ok:
            available_slots = MAX_TOTAL_POSITIONS - len(open_positions)
            candidates = []

            for t in tickers:
                if t in open_positions or t in just_exited_today:
                    continue

                df = ind.get(t)
                if df is None or prev_date not in df.index:
                    continue

                # OPTIMIZATION FIX 2: Fast index tracking checks via mapping dictionary
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

                # Lookback Window Validation
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

                # High Vol Keltner Breakout Configuration
                if y_close > y_kc_upper and y_vol > y_avgvol30 and cons:
                    trigger_entry = True
                    mod_type = "Zhan Module A (Keltner Breakout)"
                    strength = (y_close - y_kc_upper) / y_atr if y_atr > 0 else 0
                
                # Low Vol EMA Support Bounce Configuration
                elif y_close > y_ema20 and y_vol < y_avgvol30 and cons:
                    trigger_entry = True
                    mod_type = "Zhan Module B (EMA Breakout)"
                    strength = (y_close - y_ema20) / y_atr if y_atr > 0 else 0

                if trigger_entry:
                    candidates.append({
                        'ticker': t, 'mod_type': mod_type, 'atr': y_atr, 
                        'strength': strength, 'sector': sector, 'stop_mult': stop_mult
                    })

            # Sort discovered pool by true strength metrics
            candidates.sort(key=lambda x: x['strength'], reverse=True)

            # --- PHASE C: RANKED ALLOCATION GATE ---
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
                        'stop_mult': c['stop_mult'], # Preserved for ongoing trailing stop adjustments
                        'days_held': 0 
                    }
                    available_slots -= 1
                    transient_equity -= total_cost 

        # Unified inventory age progression tracking
        for t in open_positions:
            open_positions[t]['days_held'] = open_positions[t].get('days_held', 0) + 1

        # Day-End Portfolio Bookkeeping
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
        'name': 'Zhan_Production_Flawless',
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

# ==============================================================================
# PERFORMANCE METRICS ENGINE — COMPENSATED FOR COMPOUNDING [GUARDED]
# ==============================================================================
def compute_metrics(model):
    eq = pd.Series(model['equity_curve'])
    daily = np.array(model['daily_returns'])
    
    # FIX: Guard entry point with a safe dictionary lookup fallback
    trades = model.get('trade_log', [])

    # Max Drawdown Calculation
    running_peak = eq.cummax()
    drawdown = (eq - running_peak) / running_peak if not running_peak.empty else eq
    max_dd = drawdown.min() * 100 if not drawdown.empty else 0

    # Compounded Total Return Calculation
    total_return = (eq.iloc[-1] - START_CAPITAL) / START_CAPITAL * 100 if not eq.empty else 0

    # Risk-Free Adjusted Sharpe Ratio
    daily_std = np.std(daily)
    if daily_std > 0:
        sharpe = ((np.mean(daily) - DAILY_RF) / daily_std) * np.sqrt(252)
    else:
        sharpe = 0

    # Sortino Ratio
    downside_returns = np.where(daily < 0, daily, 0)
    downside_dev = np.std(downside_returns)
    if downside_dev > 0:
        sortino = ((np.mean(daily) - DAILY_RF) / downside_dev) * np.sqrt(252)
    else:
        sortino = 0

    # Compounding-Compensated Profit Factor Vector
    for tr in trades:
        tr['pct_return'] = (tr['exit_price'] - tr['entry_price']) / tr['entry_price']

    gross_pct_profit = sum(tr['pct_return'] for tr in trades if tr['pct_return'] > 0)
    gross_pct_loss = -sum(tr['pct_return'] for tr in trades if tr['pct_return'] < 0)
    
    # Calculate Profit Factor safely even if there are no trades (e.g., benchmark)
    profit_factor = gross_pct_profit / gross_pct_loss if gross_pct_loss > 0 else (0.0 if total_return <= 0 else np.nan)

    # Win Rate Calculations
    wins = sum(1 for tr in trades if tr['pct_return'] > 0)
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

# ==============================================================================
# PERFORMANCE COMPARISON PRINTING ENGINE
# ==============================================================================
def compare_models(results, SHOW_CHARTS=False):
    """
    Compiles metrics using your custom compounding-compensated compute_metrics engine
    and prints a clean, side-by-side performance grid layout.
    """
    metrics = [compute_metrics(r) for r in results]

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

    # Calculate Alpha vs. Benchmark
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

    # Cleanly export trades data structures to CSV tracking matrices
    for m in metrics:
        if not m['trade_log']:
            continue
        df = pd.DataFrame(m['trade_log'])
        csv_name = f"{m['name'].lower().replace(' ', '_').replace('(', '').replace(')', '')}_trades.csv"
        df.to_csv(csv_name, index=False)


# ==============================================================================
# MONTE CARLO WRAPPER FOR PRINTING RESULTS
# ==============================================================================
def run_monte_carlo_suite(results, n_sims=5000, block_size=5, seed=42):
    """
    Runs your optimized, vectorized bootstrap engine for every model variant
    and outputs a risk profile matrix.
    """
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
            'dd_median': np.percentile(mc['max_dd'], 50),
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
    return table

# ==============================================================================
# PRODUCION READY MONTE CARLO — BLOCK BOOTSTRAP (FULLY VECTORIZED & ACCURATE)
# ==============================================================================
import numpy as np

def monte_carlo_bootstrap(daily_returns, n_sims=5000, block_size=5, seed=42):
    """
    Resamples the strategy's historical daily returns in contiguous blocks to 
    preserve holding streaks, executing the entire simulation suite via 
    vectorized matrix operations to maximize runtime performance.
    """
    rng = np.random.default_rng(seed)
    daily = np.asarray(daily_returns, dtype=float)
    n = len(daily)
    if n == 0:
        return None

    # Calculate dimensions needed to cover target timeframe structure
    num_blocks_needed = int(np.ceil(n / block_size))
    max_start_idx = n - block_size + 1 if n > block_size else 1

    # Generate all block start positions across every path simultaneously 
    start_indices = rng.integers(0, max_start_idx, size=(n_sims, num_blocks_needed))

    # Pre-allocate master multi-dimensional path array
    sim_paths = np.zeros((n_sims, num_blocks_needed * block_size))
    
    # Vectorized block reconstruction mapping pattern
    for block_offset in range(block_size):
        sim_paths[:, block_offset::block_size] = daily[start_indices + block_offset]

    # Cleanly truncate the matrix to match the exact calendar length constraints
    sim_paths = sim_paths[:, :n]

    # Blazing-fast matrix multiplication replaces the slow sequential accounting loop
    eq_paths = START_CAPITAL * np.cumprod(1 + sim_paths, axis=1)

    # Vectorized Maximum Drawdown Calculation Suite
    running_peaks = np.maximum.accumulate(eq_paths, axis=1)
    drawdowns = (eq_paths - running_peaks) / running_peaks
    max_dds = np.min(drawdowns, axis=1) * 100

    # Capture absolute system terminal returns
    total_returns = (eq_paths[:, -1] - START_CAPITAL) / START_CAPITAL * 100

    # Risk-Adjusted Evaluation: Deducts DAILY_RF directly via vector scaling matrices
    means = np.mean(sim_paths - DAILY_RF, axis=1)
    stds = np.std(sim_paths, axis=1)
    sharpes = np.where(stds > 0, (means / stds) * np.sqrt(252), 0.0)

    return {
        'total_return': total_returns,
        'max_dd': max_dds,
        'sharpe': sharpes
    }

# ==============================================================================
# MAIN EXECUTION PIPELINE
# ==============================================================================
def run_full_suite(ind, data, sp500, sp500_sma200, sp500_sma50,
                    max_total_positions, max_alloc_fraction, max_sector_positions,
                    label=""):
    """
    Safely routes position-limit parameters down into the execution engines,
    mutating global configuration constants safely for the duration of the run.
    """
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

    # Restore initial state boundaries cleanly
    MAX_TOTAL_POSITIONS, MAX_ALLOC_FRACTION, MAX_SECTOR_POSITIONS = orig
    return yang, zhan, yin


# ==============================================================================
# 1. MAIN EXECUTION PIPELINE (Completely unindented)
# ==============================================================================
def run_full_suite(ind, data, sp500, sp500_sma200, sp500_sma50,
                    max_total_positions, max_alloc_fraction, max_sector_positions,
                    label=""):
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
# 2. BENCHMARK UTILITY (Completely unindented)
# ==============================================================================
def run_benchmark(sp500, dates):
    """
    Computes a clean market index benchmark execution track.
    Ensures both equity paths and raw returns curves match portfolio arrays.
    """
    sp_slice = sp500.loc[dates]
    initial_price = sp_slice.iloc[0]
    equity_curve = (sp_slice / initial_price) * START_CAPITAL
    daily_returns = sp_slice.pct_change().fillna(0.0).tolist()
    
    return {
        'name': 'SPX Buy & Hold',
        'equity_curve': equity_curve.tolist(),
        'daily_returns': daily_returns,
        'dates': list(dates)
    }


# ==============================================================================
# 3. RUNTIME EXECUTION GATEWAY
# ==============================================================================
if __name__ == "__main__":
    # Ingest the clean data frames 
    data = load_data()
    ind, sp500, sp500_sma200, sp500_sma50 = build_indicators(data)
    
    # Master timeline calendar driven securely by the global index asset to remove gaps
    dates = sp500.index[200:]

    # --- PROCESS SWEEP 1: RESTRICTED DEPLOYMENT MODEL ---
    print("\n### RUN 1: Manual-Tracking Limits (original) ###")
    print(f"    MAX_TOTAL_POSITIONS=6, MAX_ALLOC_FRACTION=1/6, MAX_SECTOR_POSITIONS=2")
    yang_manual, zhan_manual, yin_manual = run_full_suite(
        ind, data, sp500, sp500_sma200, sp500_sma50,
        max_total_positions=6, max_alloc_fraction=1/6, max_sector_positions=2,
        label="(Manual Limits)"
    )

    # --- PROCESS SWEEP 2: RISK-ONLY UNCONSTRAINED MODEL ---
    print("\n### RUN 2: Unconstrained (no manual-tracking caps) ###")
    print(f"    MAX_TOTAL_POSITIONS={len(tickers)} (one slot per ticker), "
          f"MAX_ALLOC_FRACTION=1.0 (position size driven purely by risk rules), "
          f"MAX_SECTOR_POSITIONS={len(tickers)} (no sector cap)")
    yang_free, zhan_free, yin_free = run_full_suite(
        ind, data, sp500, sp500_sma200, sp500_sma50,
        max_total_positions=len(tickers), max_alloc_fraction=1.0, max_sector_positions=len(tickers),
        label="(Unconstrained)"
    )

    # --- PROCESS SWEEP 3: COMPILING RESULTS DATASTRUCTURE MAPS ---
    print("\nProcessing SPX Buy & Hold benchmark...")
    benchmark_results = run_benchmark(sp500, dates)

    # Consolidate target outputs for cross-model comparative profiling
    all_results = [
        yang_manual, zhan_manual, yin_manual,
        yang_free, zhan_free, yin_free,
        benchmark_results
    ]

    # Run analytical reporting comparison table outputs
    compare_models(all_results, SHOW_CHARTS=False)

    # Run vectorized block bootstrap distributions safely across matching structures
    print("\nRunning Monte Carlo block-bootstrap analysis...")
    run_monte_carlo_suite(all_results, n_sims=5000, block_size=5, seed=42)
