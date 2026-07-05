

import pandas as pd
import numpy as np
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
    # Updated to historical range up to your July 3 backtest benchmark cut-off
    data = yf.download(tickers + ['^GSPC'], start="2023-01-01", end="2026-07-03")
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

        # ATR(14)
        tr1 = df['High'] - df['Low']
        tr2 = (df['High'] - df['Close'].shift(1)).abs()
        tr3 = (df['Low'] - df['Close'].shift(1)).abs()
        df['ATR14'] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1).rolling(14).mean()

        # RSI(2)
        delta = df['Close'].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(2).mean()
        avg_loss = loss.rolling(2).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        df['RSI2'] = 100 - (100 / (1 + rs))

        # Keltner Channels (FIXED: Multiplier set to 1.5 to align with Zhan's Volatility Expansion model rules)
        df['KC_MID'] = df['EMA20']
        df['KC_UPPER'] = df['EMA20'] + 1.5 * df['ATR14']
        df['KC_LOWER'] = df['EMA20'] - 1.5 * df['ATR14']

        ind[t] = df

    # Market filter
    sp500 = close['^GSPC']
    sp500_sma200 = sp500.rolling(200).mean()

    return ind, sp500, sp500_sma200

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
    if open_price < stop_price:
        return open_price
    return close_price

# ==========================================
# FIXED HOLE 3: FRACTIONAL-FRICTION SIZING MATRIX
# ==========================================
def position_size(entry_price, stop_price, equity):
    # Adjust the baseline entry price for friction before verifying distances
    friction_entry = apply_entry_friction(entry_price)
    risk_dist = friction_entry - stop_price
    if risk_dist <= 0:
        return 0

    # Subtract the flat $1 commission from allocated risk capital to protect 1% ceiling
    risk_dollars = (equity * RISK_PER_TRADE) - COMMISSION
    shares_risk = risk_dollars / risk_dist

    # Apply identical parameter friction to capital allocation constraints
    max_alloc = (equity * MAX_ALLOC_FRACTION) - COMMISSION
    shares_alloc = max_alloc / friction_entry

    return int(min(shares_risk, shares_alloc))

# ==========================================
# SHARED UTILITY: SECTOR CAP CHECK
# ==========================================
def sector_cap_reached(open_positions, sector):
    count = sum(1 for p in open_positions.values() if p['sector'] == sector)
    return count >= MAX_SECTOR_POSITIONS

    
# MODEL A — YANG (Trend Following) - RE-ARCHITECTED
# ==========================================
def run_yang(ind, data, sp500, sp500_sma200):
    cash = START_CAPITAL
    equity = START_CAPITAL
    open_positions = {}
    trade_log = []
    equity_curve = []
    daily_returns = []

    # Start after indicators mature
    dates = ind[tickers[0]].index[200:]  

    for i in range(1, len(dates)):
        date = dates[i]
        prev_date = dates[i-1]

        # Apply cash drag to uninvested cash before evaluating allocations
        cash = apply_cash_drag(cash)

        # Market filter evaluated on completed historical day (T-1)
        market_ok = sp500.loc[prev_date] > sp500_sma200.loc[prev_date]

        # ============================
        # A. FIXED HOLE 1: NEXT-DAY OPEN EXITS
        # ============================
        to_close = []
        for t, pos in open_positions.items():
            df = ind[t]
            # Signals are generated on completed data from the prior close (T-1)
            prev_close = df.loc[prev_date, 'Close']
            prev_sma50 = df.loc[prev_date, 'SMA50']

            # If rules were broken yesterday, mark for liquidation this morning
            if prev_close < prev_sma50 or prev_close <= pos['stop']:
                to_close.append(t)

        for t in to_close:
            pos = open_positions.pop(t)
            # Liquidate at today's true opening market print (T Open)
            open_price = data['Open'][t].loc[date]
            
            # Apply real-world exit friction and commissions
            exit_price = gap_down_exit(open_price, pos['stop'], open_price)
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

        # ============================
        # B. FIXED HOLE 2: MULTI-ENTRY CIRCUIT BREAKER
        # ============================
        if market_ok:
            # Calculate exactly how many asset slots are open today
            available_slots = MAX_TOTAL_POSITIONS - len(open_positions)
            
            for t in tickers:
                # Trigger circuit breaker if portfolio is at maximum capacity
                if available_slots <= 0:
                    break
                    
                if t in open_positions:
                    continue

                df = ind[t]

                # Extract signals strictly from yesterday's completed bars (T-1)
                y_close = df.loc[prev_date, 'Close']
                y_sma50 = df.loc[prev_date, 'SMA50']
                y_prior_high = df.loc[prev_date, 'Prior20High']
                y_vol = df.loc[prev_date, 'Vol']
                y_avgvol30 = df.loc[prev_date, 'AvgVol30']
                y_atr = df.loc[prev_date, 'ATR14']

                sector = sectors[t]
                if sector_cap_reached(open_positions, sector):
                    continue

                # Entry Module A (High Volume Breakout)
                signal_a = (
                    y_close > y_sma50 and
                    y_vol > y_avgvol30 and
                    y_close > y_prior_high
                )

                # Entry Module B (Low Volume Pullback)
                signal_b = (
                    y_close > y_sma50 and
                    y_vol < y_avgvol30 and
                    y_close <= (df.loc[prev_date, 'SMA20'] + 2 * y_atr)
                )

                if signal_a or signal_b:
                    # Execute entry at today's opening bell print (T Open)
                    raw_open = data['Open'][t].loc[date]
                    
                    # Calculate position stop distance based on raw open price
                    stop = raw_open - (2 * y_atr)
                    
                    # Position sizing handles inner entry friction and commission caps
                    shares = position_size(raw_open, stop, equity)
                    
                    # Re-adjust entry price with friction for proper account balance ledger
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
                        # Decrement available portfolio allocation slots immediately
                        available_slots -= 1

        # ============================
        # C. EQUITY UPDATE
        # ============================
        # Real-time portfolio tracking using today's completed closing print (T Close)
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
        'daily_returns': daily_returns
    }

    

# ==========================================
# MODEL C — YIN (Mean Reversion) - RE-ARCHITECTED
# ==========================================
def run_yin(ind, data, sp500, sp500_sma200):
    cash = START_CAPITAL
    equity = START_CAPITAL
    open_positions = {}
    trade_log = []
    equity_curve = []
    daily_returns = []

    # Start after indicators mature
    dates = ind[tickers[0]].index[200:]

    for i in range(1, len(dates)):
        date = dates[i]
        prev_date = dates[i-1]

        # Apply cash drag to uninvested cash before evaluating allocations
        cash = apply_cash_drag(cash)

        # Market filter evaluated on completed historical day (T-1)
        market_ok = sp500.loc[prev_date] > sp500_sma200.loc[prev_date]

        # ============================
        # A. FIXED HOLE 1 & TIME-STOP: NEXT-DAY OPEN EXITS
        # ============================
        to_close = []
        for t, pos in open_positions.items():
            df = ind[t]
            # Exits are checked using completed data from yesterday's close (T-1)
            prev_close = df.loc[prev_date, 'Close']
            prev_sma5 = df.loc[prev_date, 'SMA5']

            # 1. FIXED TIME-STOP: Check true accumulated trading days
            if pos['days_held'] >= 4:
                to_close.append(t)
            # 2. STANDARD EXITS: Check price milestones against yesterday's indicators
            elif prev_close > prev_sma5 or prev_close <= pos['stop']:
                to_close.append(t)

        for t in to_close:
            pos = open_positions.pop(t)
            
            # Liquidate at today's true opening market print (T Open)
            open_price = data['Open'][t].loc[date]
            
            # Apply real-world exit friction and commissions
            exit_price = gap_down_exit(open_price, pos['stop'], open_price)
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

        # ============================
        # B. FIXED HOLE 2 & 4: MULTI-ENTRY CIRCUIT BREAKER & IF/ELIF CHAIN
        # ============================
        if market_ok:
            # Calculate exactly how many asset slots are open today
            available_slots = MAX_TOTAL_POSITIONS - len(open_positions)

            for t in tickers:
                # Trigger circuit breaker if portfolio is at maximum capacity (Hole 2)
                if available_slots <= 0:
                    break
                    
                if t in open_positions:
                    continue

                df = ind[t]

                # Extract signals strictly from yesterday's completed bars (T-1)
                y_close = df.loc[prev_date, 'Close']
                y_rsi2 = df.loc[prev_date, 'RSI2']
                y_sma200 = df.loc[prev_date, 'SMA200']
                y_atr = df.loc[prev_date, 'ATR14']

                sector = sectors[t]
                if sector_cap_reached(open_positions, sector):
                    continue

                # RE-ARCHITECTED IF/ELIF CHAIN (FIXED HOLE 4): Prevents asset duplication
                is_triggered = False
                
                if y_close > y_sma200 and y_rsi2 < 5:
                    # Module B (Deep Pullback) takes absolute priority
                    is_triggered = True
                elif y_close > y_sma200 and y_rsi2 < 10:
                    # Module A (Standard Pullback) triggers only if RSI is between 5 and 10
                    is_triggered = True

                if is_triggered:
                    # Execute entry at today's opening bell print (T Open)
                    raw_open = data['Open'][t].loc[date]
                    
                    # Calculate position stop distance based on raw open price
                    stop = raw_open - (1.0 * y_atr)
                    
                    # Position sizing handles inner entry friction and commission caps (Hole 3)
                    shares = position_size(raw_open, stop, equity)

                    # Re-adjust entry price with friction for proper account balance ledger
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
                            'days_held': 0  # Initialize trading day tracking counter
                        }
                        # Decrement available portfolio allocation slots immediately
                        available_slots -= 1

        # Increment trading days held for all remaining open positions
        for t in open_positions:
            open_positions[t]['days_held'] += 1

        # ============================
        # C. EQUITY UPDATE
        # ============================
        # Real-time portfolio tracking using today's completed closing print (T Close)
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
        'daily_returns': daily_returns
    }



# ==========================================
# MODEL B — ZHAN (Volatility Expansion) - RE-ARCHITECTED
# ==========================================
def run_zhan(ind, data, sp500, sp500_sma200):
    cash = START_CAPITAL
    equity = START_CAPITAL
    open_positions = {}
    trade_log = []
    equity_curve = []
    daily_returns = []

    # Start after indicators mature
    dates = ind[tickers[0]].index[200:]

    for i in range(1, len(dates)):
        date = dates[i]
        prev_date = dates[i-1]

        # Apply cash drag to uninvested cash before evaluating allocations
        cash = apply_cash_drag(cash)

        # Market filter evaluated on completed historical day (T-1)
        market_ok = sp500.loc[prev_date] > sp500_sma200.loc[prev_date]

        # ============================
        # A. FIXED HOLE 1: NEXT-DAY OPEN EXITS
        # ============================
        to_close = []
        for t, pos in open_positions.items():
            df = ind[t]
            # Exits are checked using completed data from yesterday's close (T-1)
            prev_close = df.loc[prev_date, 'Close']
            prev_kc_lower = df.loc[prev_date, 'KC_LOWER']

            # If rules were broken yesterday, mark for liquidation this morning
            if prev_close < prev_kc_lower or prev_close <= pos['stop']:
                to_close.append(t)

        for t in to_close:
            pos = open_positions.pop(t)
            
            # Liquidate at today's true opening market print (T Open)
            open_price = data['Open'][t].loc[date]
            
            # Apply real-world exit friction and commissions
            exit_price = gap_down_exit(open_price, pos['stop'], open_price)
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

        # ============================
        # B. FIXED HOLE 2 & 5: MULTI-ENTRY CIRCUIT BREAKER & LOOKBACK INDEX
        # ============================
        if market_ok:
            # Calculate exactly how many asset slots are open today
            available_slots = MAX_TOTAL_POSITIONS - len(open_positions)

            for t in tickers:
                # Trigger circuit breaker if portfolio is at maximum capacity (Hole 2)
                if available_slots <= 0:
                    break
                    
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

                # FIXED HOLE 5: Standardized index arrays to include yesterday's close (curr_idx)
                # Verifies tight consolidation inside the 1.5x ATR Keltner bounds over prior 3 consecutive sessions
                cons = (
                    df.iloc[curr_idx]['Close'] < df.iloc[curr_idx]['KC_UPPER'] and
                    df.iloc[curr_idx]['Close'] > df.iloc[curr_idx]['KC_LOWER'] and
                    df.iloc[curr_idx-1]['Close'] < df.iloc[curr_idx-1]['KC_UPPER'] and
                    df.iloc[curr_idx-1]['Close'] > df.iloc[curr_idx-1]['KC_LOWER'] and
                    df.iloc[curr_idx-2]['Close'] < df.iloc[curr_idx-2]['KC_UPPER'] and
                    df.iloc[curr_idx-2]['Close'] > df.iloc[curr_idx-2]['KC_LOWER']
                )

                # Entry Module A (High Volume Keltner Breakout)
                signal_a = (y_close > y_kc_upper and y_vol > y_avgvol30 and cons)
                
                # Entry Module B (Low Volume EMA Pullback State)
                signal_b = (y_close > y_ema20 and y_vol < y_avgvol30)

                if signal_a or signal_b:
                    # Execute entry at today's opening bell print (T Open)
                    raw_open = data['Open'][t].loc[date]
                    
                    # Calculate position stop distance based on raw open price (1.5x ATR)
                    stop = raw_open - (1.5 * y_atr)
                    
                    # Position sizing handles inner entry friction and commission caps (Hole 3)
                    shares = position_size(raw_open, stop, equity)
                    
                    # Re-adjust entry price with friction for proper account balance ledger
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
                        # Decrement available portfolio allocation slots immediately
                        available_slots -= 1

        # ============================
        # C. EQUITY UPDATE
        # ============================
        # Real-time portfolio tracking using today's completed closing print (T Close)
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
        'daily_returns': daily_returns
    }



# ==========================================
# FIXED PERFORMANCE METRICS ENGINE
# ==========================================
def compute_metrics(model):
    eq = pd.Series(model['equity_curve'])
    daily = np.array(model['daily_returns'])
    trades = model['trade_log']

    # Running peak calculation for Drawdown curves
    running_peak = eq.cummax()
    drawdown = (eq - running_peak) / running_peak
    max_dd = drawdown.min() * 100

    # Total Return calculation
    total_return = (eq.iloc[-1] - START_CAPITAL) / START_CAPITAL * 100

    # Standardized Sharpe Ratio Calculation
    daily_std = np.std(daily)
    if daily_std != 0:
        sharpe = (np.mean(daily) / daily_std) * np.sqrt(252)
    else:
        sharpe = 0

    # CORRECTED INSTITUTIONAL SORTINO DEV MATH
    # Set all positive return segments to 0 instead of dropping them from array length
    downside_returns = np.where(daily < 0, daily, 0)
    downside_dev = np.std(downside_returns)
    
    if downside_dev > 0:
        sortino = (np.mean(daily) / downside_dev) * np.sqrt(252)
    else:
        sortino = 0

    # Profit Factor Engine
    gross_profit = sum(tr['pnl'] for tr in trades if tr['pnl'] > 0)
    gross_loss = -sum(tr['pnl'] for tr in trades if tr['pnl'] < 0)
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.nan

    # FIXED TRACKING MODULE: Capture total trades including flat/scratch executions
    wins = sum(1 for tr in trades if tr['pnl'] > 0)
    total_trades = len(trades)
    win_rate = (wins / total_trades * 100) if total_trades > 0 else 0

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
# COMPARISON DASHBOARD & GRAPHICS CONTROLLER
# ==========================================
def compare_models(results, SHOW_CHARTS=False):
    # Process calculations through fixed metrics engine
    metrics = [compute_metrics(r) for r in results]

    if SHOW_CHARTS:
        # ==========================================
        # A. EQUITY CURVE PLOT
        # ==========================================
        plt.figure(figsize=(12,6))
        for m in metrics:
            plt.plot(m['equity_curve'], label=m['name'])
        plt.title("Equity Curves — Yang vs Zhan vs Yin")
        plt.legend()
        plt.grid(True)
        plt.show()

        # ==========================================
        # B. DRAWDOWN CHART
        # ==========================================
        plt.figure(figsize=(12,4))
        for m in metrics:
            plt.plot(m['drawdown'] * 100, label=m['name'])
        plt.title("Drawdown Comparison (%)")
        plt.legend()
        plt.grid(True)
        plt.show()

    # ==========================================
    # C. PERFORMANCE TABLE
    # ==========================================
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
    print("MODEL PERFORMANCE COMPARISON")
    print("==============================")
    print(table.to_string(index=False))

    # ==========================================
    # D. STRATEGY RANKING
    # ==========================================
    print("\n==============================")
    print("STRATEGY RANKING (by Total Return)")
    print("==============================")
    ranked = table.sort_values(by='Total Return %', ascending=False)
    print(ranked[['Model', 'Total Return %']].to_string(index=False))

    # ==========================================
    # E. CSV EXPORTS
    # ==========================================
    for m in metrics:
        df = pd.DataFrame(m['trade_log'])
        df.to_csv(f"{m['name'].lower()}_trades.csv", index=False)
        print(f"Exported {m['name']} trades → {m['name'].lower()}_trades.csv")

# ==========================================
# MAIN EXECUTION PIPELINE
# ==========================================
if __name__ == "__main__":
    data = load_data()
    ind, sp500, sp500_sma200 = build_indicators(data)

    print("\nProcessing Model A (Yang)...")
    yang_results = run_yang(ind, data, sp500, sp500_sma200)
    
    print("Processing Model B (Zhan)...")
    zhan_results = run_zhan(ind, data, sp500, sp500_sma200)
    
    print("Processing Model C (Yin)...")
    yin_results = run_yin(ind, data, sp500, sp500_sma200)

    # Set SHOW_CHARTS=False to run instantly without Chromebook browser lag
    compare_models([yang_results, zhan_results, yin_results], SHOW_CHARTS=False)
