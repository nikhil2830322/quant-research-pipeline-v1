import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime

# ==============================================================================
# CONFIGURATION & LEDGER (UPDATE BALANCES HERE EVERY AFTERNOON AFTER 3:15 PM)
# ==============================================================================
YANG_CASH, YANG_POSITIONS = 100000.00, {}
ZHAN_CASH, ZHAN_POSITIONS = 100000.00, {}
YIN_CASH, YIN_POSITIONS = 100000.00, {}

START_CAPITAL = 100000
RISK_PER_TRADE = 0.01
MAX_SECTOR_POSITIONS = 2
MAX_TOTAL_POSITIONS = 6
MAX_ALLOC_FRACTION = 1/6
SLIPPAGE_ENTRY = 0.0005
SLIPPAGE_EXIT = 0.0005
COMMISSION = 1.0
RISK_FREE_ANNUAL = 0.045
DAILY_RF = RISK_FREE_ANNUAL / 252

TICKERS = ['MSFT','AMZN','JPM','UNH','XOM','WMT','NVDA','AMD','META','TSLA']
SECTORS = {
    'MSFT':'Tech','NVDA':'Tech','AMD':'Tech','META':'Tech',
    'AMZN':'Consumer','TSLA':'Consumer','WMT':'Consumer',
    'JPM':'Finance','UNH':'Healthcare','XOM':'Energy'
}
# ==============================================================================
# FRICTION PIPELINES
# ==============================================================================
def apply_entry_friction(price): 
    return price * (1 + SLIPPAGE_ENTRY)

def apply_exit_friction(price): 
    return price * (1 - SLIPPAGE_EXIT)

def sector_cap_reached(open_positions, sector):
    count = sum(1 for p in open_positions.values() if p['sector'] == sector)
    return count >= MAX_SECTOR_POSITIONS

# ==============================================================================
# MAIN DAILY SIGNAL MONITORING PLATFORM
# ==============================================================================
def check_live_signals():
    print("==============================================================================")
    print(f"LIVE TOURNAMENT EXECUTION DESK | RUNTIME: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} CDT")
    print("==============================================================================\n")
    
    # Downloads settled data frames post-market close
    data = yf.download(TICKERS + ['^GSPC'], period="250d", progress=False)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.swaplevel(0, 1)

    spy_close = data['^GSPC']['Close'].iloc[-1]
    spy_sma200 = data['^GSPC']['Close'].rolling(200).mean().iloc[-1]
    market_ok = spy_close > spy_sma200
    
    print(f"S&P 500 Index Close: {spy_close:.2f} | 200 SMA Baseline: {spy_sma200:.2f}")
    print(f"Primary Regime Filter: {'[PASS] - Entries Allowed' if market_ok else '[FAIL] - Entry Blocks Active'}\n")

    systems = [
        {'name': 'MODEL A (YANG)', 'cash': YANG_CASH, 'positions': YANG_POSITIONS},
        {'name': 'MODEL B (ZHAN)', 'cash': ZHAN_CASH, 'positions': ZHAN_POSITIONS},
        {'name': 'MODEL C (YIN)', 'cash': YIN_CASH, 'positions': YIN_POSITIONS}
    ]
    for sys in systems:
        print(f"--- ACTIVE EXITS & ORDERS FOR {sys['name']} ---")
        open_val = sum(pos['shares'] * data[t]['Close'].iloc[-1] for t, pos in sys['positions'].items())
        current_cash = sys['cash'] * (1 + DAILY_RF)
        total_equity = current_cash + open_val
        available_slots = MAX_TOTAL_POSITIONS - len(sys['positions'])
        
        print(f"  Adjusted Cash Account: ${current_cash:.2f} | Net Portfolio Equity: ${total_equity:.2f}")
        print(f"  Available Portfolio Allocation Slots: {available_slots} / {MAX_TOTAL_POSITIONS}")

        # --- A. LIQUIDATION LOGIC ---
        has_exits = False
        for t, pos in sys['positions'].items():
            df = data[t]
            close_spot = df['Close'].iloc[-1]
            trigger_exit, reason = False, ""

            tr = pd.concat([df['High']-df['Low'], (df['High']-df['Close'].shift(1)).abs(), (df['Low']-df['Close'].shift(1)).abs()], axis=1).max(axis=1)
            atr14 = tr.rolling(14).mean().iloc[-1]

            if sys['name'] == 'MODEL A (YANG)' and (close_spot < df['Close'].rolling(50).mean().iloc[-1] or close_spot <= pos['stop']):
                trigger_exit, reason = True, "SMA50 Trend Breach or Stop-Loss"
            elif sys['name'] == 'MODEL B (ZHAN)':
                ema20 = df['Close'].ewm(span=20).mean().iloc[-1]
                if close_spot < (ema20 - 1.5 * atr14) or close_spot <= pos['stop']:
                    trigger_exit, reason = True, "1.5x Keltner Lower Boundary or Stop-Loss"
            elif sys['name'] == 'MODEL C (YIN)' and (pos['days_held'] >= 4 or close_spot > df['Close'].rolling(5).mean().iloc[-1] or close_spot <= pos['stop']):
                trigger_exit, reason = True, "Time-Stop / Mean Reversion SMA5 or Stop-Loss"

            if trigger_exit:
                print(f"    🚨 [LIQUIDATE] Ticker: {t} | Reason: {reason} | ROUTE: MOC (Record in Ledger at Today's Close)")
                has_exits = True
        if not has_exits:
            print("    No open positions satisfy liquidation rules today.")
        
        # --- B. ENTRY LOGIC ---
        print("  Active Allocation Triggers:")
        if market_ok and available_slots > 0:
            for t in TICKERS:
                if t in sys['positions'] or sector_cap_reached(sys['positions'], SECTORS[t]): 
                    continue
                
                df = data[t]
                close_spot, vol_spot = df['Close'].iloc[-1], df['Volume'].iloc[-1]
                sma50 = df['Close'].rolling(50).mean().iloc[-1]
                sma20 = df['Close'].rolling(20).mean().iloc[-1]
                ema20 = df['Close'].ewm(span=20).mean().iloc[-1]
                sma200 = df['Close'].rolling(200).mean().iloc[-1]
                avgvol30 = df['Volume'].rolling(30).mean().iloc[-1]
                prior_20high = df['High'].shift(1).rolling(20).max().iloc[-1]
                tr = pd.concat([df['High']-df['Low'], (df['High']-df['Close'].shift(1)).abs(), (df['Low']-df['Close'].shift(1)).abs()], axis=1).max(axis=1)
                atr14 = tr.rolling(14).mean().iloc[-1]

                trigger_entry, mod_type, stop_price = False, "", 0.0

                if sys['name'] == 'MODEL A (YANG)':
                    if close_spot > sma50 and vol_spot > avgvol30 and close_spot > prior_20high:
                        trigger_entry, mod_type, stop_price = True, "Yang Module A (High Vol Breakout)", close_spot - (2 * atr14)
                    elif close_spot > sma50 and vol_spot < avgvol30 and close_spot <= (sma20 + 2 * atr14):
                        trigger_entry, mod_type, stop_price = True, "Yang Module B (Low Vol Pullback)", close_spot - (2 * atr14)

                elif sys['name'] == 'MODEL B (ZHAN)':
                    kc_upper = ema20 + 1.5 * atr14
                    cons = all((df['Close'].iloc[-i] < (df['Close'].ewm(span=20).mean().iloc[-i] + 1.5 * tr.rolling(14).mean().iloc[-i]) and \
                                df['Close'].iloc[-i] > (df['Close'].ewm(span=20).mean().iloc[-i] - 1.5 * tr.rolling(14).mean().iloc[-i])) for i in range(2,5))
                    if close_spot > kc_upper and vol_spot > avgvol30 and cons:
                        trigger_entry, mod_type, stop_price = True, "Zhan Module A (Keltner Breakout)", close_spot - (1.5 * atr14)
                    elif close_spot > ema20 and vol_spot < avgvol30 and cons:
                        trigger_entry, mod_type, stop_price = True, "Zhan Module B (EMA Breakout)", close_spot - (1.5 * atr14)

                elif sys['name'] == 'MODEL C (YIN)':
                    delta = df['Close'].diff()
                    gain, loss = delta.clip(lower=0), -delta.clip(upper=0)
                    rsi2 = (100 - (100 / (1 + (gain.rolling(2).mean() / loss.rolling(2).mean().replace(0, np.nan))))).iloc[-1]
                    if close_spot > sma200:
                        if rsi2 < 5:
                            trigger_entry, mod_type, stop_price = True, "Yin Module B (Deep RSI < 5)", close_spot - (1.0 * atr14)
                        elif rsi2 < 10:
                            trigger_entry, mod_type, stop_price = True, "Yin Module A (Standard RSI < 10)", close_spot - (1.0 * atr14)

                if trigger_entry and available_slots > 0:
                    friction_entry = apply_entry_friction(close_spot)
                    risk_dist = friction_entry - stop_price
                    if risk_dist > 0:
                        risk_cap = (total_equity * RISK_PER_TRADE) - COMMISSION
                        shares_risk = risk_cap / risk_dist
                        max_alloc = (total_equity * MAX_ALLOC_FRACTION) - COMMISSION
                        shares_alloc = max_alloc / friction_entry
                        final_shares = int(min(shares_risk, shares_alloc))
                        if final_shares > 0:
                            print(f"    ⭐ [ENTRY] Ticker: {t} | Setup: {mod_type} | Volume: {final_shares} Shares | Entry Price: ${friction_entry:.2f} | Stop Base: ${stop_price:.2f} | ROUTE: MOC")
                            available_slots -= 1

        else:
            if not market_ok: 
                print("    All entry modules blocked by primary S&P 500 Market Filter.")
            if available_slots <= 0: 
                print("    Entry modules locked. Portfolio allocation is at maximum capacity.")
        print("------------------------------------------------------------------------------\n")

if __name__ == "__main__":
    check_live_signals()
