import pandas as pd
import numpy as np
import yfinance as yf

# ==========================================
# 1. SETUP PARAMETERS & UNIVERSE
# ==========================================
START_PORTFOLIO = 100000.0
RISK_PER_TRADE = 0.01
MAX_SECTOR_POSITIONS = 2
MAX_TOTAL_POSITIONS = 6
MAX_ALLOCATION_FRACTION = 1/6

tickers = ['MSFT','AMZN','JPM','UNH','XOM','WMT','NVDA','AMD','META','TSLA']
sectors = {
    'MSFT':'Tech','NVDA':'Tech','AMD':'Tech','META':'Tech',
    'AMZN':'Consumer','TSLA':'Consumer','WMT':'Consumer',
    'JPM':'Finance','UNH':'Healthcare','XOM':'Energy'
}

print("Downloading historical data...")
data = yf.download(tickers + ['^GSPC'], start="2023-01-01", end="2026-07-03")
close = data['Close']
high = data['High']
low = data['Low']
vol = data['Volume']

# ==========================================
# 2. INDICATORS (NO LOOK-AHEAD)
# ==========================================
ind = {}

for t in tickers:
    df = pd.DataFrame(index=close.index)
    df['Close'] = close[t]
    df['High'] = high[t]
    df['Low'] = low[t]
    df['Vol'] = vol[t]

    df['SMA50'] = df['Close'].rolling(50).mean()
    df['SMA20'] = df['Close'].rolling(20).mean()
    df['AvgVol30'] = df['Vol'].rolling(30).mean()

    # PRIOR 20-DAY HIGH (shifted to avoid look-ahead)
    df['Prior20High'] = df['Close'].rolling(20).max().shift(1)

    # ATR(14)
    tr1 = df['High'] - df['Low']
    tr2 = (df['High'] - df['Close'].shift(1)).abs()
    tr3 = (df['Low'] - df['Close'].shift(1)).abs()
    df['ATR14'] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1).rolling(14).mean()

    ind[t] = df

# Market filter
sp500 = close['^GSPC']
sp500_sma200 = sp500.rolling(200).mean()

# ==========================================
# 3. SIMULATION ENGINE (BIAS-FREE)
# ==========================================
portfolio_value = START_PORTFOLIO
cash = START_PORTFOLIO
open_positions = {}

equity_curve = []
dates = close.index[200:]  # start after indicators mature

for i in range(1, len(dates)):
    date = dates[i]
    prev_date = dates[i-1]

    # Market filter
    if sp500.loc[prev_date] <= sp500_sma200.loc[prev_date]:
        market_ok = False
    else:
        market_ok = True

    # ============================
    # A. EXITS (today's close)
    # ============================
    to_close = []
    for t, pos in open_positions.items():
        c = ind[t].loc[date, 'Close']
        sma50 = ind[t].loc[date, 'SMA50']

        # Exit conditions
        if c < sma50 or c <= pos['stop']:
            to_close.append(t)

    for t in to_close:
        pos = open_positions.pop(t)
        exit_price = ind[t].loc[date, 'Close']
        cash += pos['shares'] * exit_price

    # ============================
    # B. ENTRIES (signals from prev_date, enter at today's open)
    # ============================
    if market_ok and len(open_positions) < MAX_TOTAL_POSITIONS:

        for t in tickers:
            if t in open_positions:
                continue

            df = ind[t]

            # Yesterday's values (signal day)
            y_close = df.loc[prev_date, 'Close']
            y_sma50 = df.loc[prev_date, 'SMA50']
            y_prior_high = df.loc[prev_date, 'Prior20High']
            y_vol = df.loc[prev_date, 'Vol']
            y_avgvol30 = df.loc[prev_date, 'AvgVol30']
            y_atr = df.loc[prev_date, 'ATR14']

            # Sector cap
            sector = sectors[t]
            sector_count = sum(1 for p in open_positions.values() if p['sector'] == sector)
            if sector_count >= MAX_SECTOR_POSITIONS:
                continue

            # Entry Module A (High Volume Breakout)
            signal_a = (y_close > y_sma50) and (y_vol > y_avgvol30) and (y_close > y_prior_high)

            # Entry Module B (Low Volume Pullback)
            signal_b = (y_close > y_sma50) and (y_vol < y_avgvol30) and (y_close <= (df.loc[prev_date,'SMA20'] + 2*y_atr))

            if signal_a or signal_b:
                # Enter at today's open
                entry_price = data['Open'][t].loc[date]
                stop = entry_price - (2 * y_atr)
                risk_dist = entry_price - stop

                if risk_dist <= 0:
                    continue

                # Risk sizing
                risk_dollars = portfolio_value * RISK_PER_TRADE
                shares_risk = risk_dollars / risk_dist

                # Allocation cap
                max_alloc = portfolio_value * MAX_ALLOCATION_FRACTION
                shares_alloc = max_alloc / entry_price

                shares = int(min(shares_risk, shares_alloc))

                if shares > 0 and cash >= shares * entry_price:
                    cash -= shares * entry_price
                    open_positions[t] = {
                        'entry': entry_price,
                        'stop': stop,
                        'shares': shares,
                        'sector': sector
                    }

    # ============================
    # C. EQUITY UPDATE
    # ============================
    open_val = sum(open_positions[t]['shares'] * ind[t].loc[date, 'Close'] for t in open_positions)
    portfolio_value = cash + open_val
    equity_curve.append(portfolio_value)

# ==========================================
# 4. SUMMARY
# ==========================================
final_return = (portfolio_value - START_PORTFOLIO) / START_PORTFOLIO * 100
print("\n==============================")
print("YANG STRATEGY BACKTEST SUMMARY")
print("==============================")
print(f"Starting Capital : ${START_PORTFOLIO:,.2f}")
print(f"Ending Capital   : ${portfolio_value:,.2f}")
print(f"Total Return     : {final_return:.2f}%")
print(f"Open Positions   : {list(open_positions.keys())}")
print("==============================")
