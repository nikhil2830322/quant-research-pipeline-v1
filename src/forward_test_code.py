import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime

# ==============================================================================
# CONFIGURATION & LEDGER
# ==============================================================================
# Update these values daily before execution
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
# FRICTION AND PRICING AUXILIARY PIPELINES
# ==============================================================================
def apply_entry_friction(price): return price * (1 + SLIPPAGE_ENTRY)
def apply_exit_friction(price): return price * (1 - SLIPPAGE_EXIT)

def sector_cap_reached(open_positions, sector):
    count = sum(1 for p in open_positions.values() if p['sector'] == sector)
    return count >= MAX_SECTOR_POSITIONS

# ==============================================================================
# MAIN DAILY SIGNAL MONITORING PLATFORM
# ==============================================================================
def check_live_signals():
    print("==============================================================================")
    print(f"LIVE TOURNAMENT EXECUTION DESK | RUNTIME: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("==============================================================================\n")
    
    data = yf.download(TICKERS + ['^GSPC'], period="250d", progress=False)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.swaplevel(0, 1)

    spy_close = data['^GSPC']['Close'].iloc[-1]
    spy_sma200 = data['^GSPC']['Close'].rolling(200).mean().iloc[-1]
    market_ok = spy_close > spy_sma200
    
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
        
        print(f"  Adjusted Ledger Cash: ${current_cash:.2f} | Net Portfolio Equity: ${total_equity:.2f}")

        # --- LIQUIDATION LOGIC ---
        has_exits = False
        for t, pos in sys['positions'].items():
            df = data[t]
            close_spot = df['Close'].iloc[-1]
            trigger_exit, reason = False, ""

            if sys['name'] == 'MODEL A (YANG)' and (close_spot < df['Close'].rolling(50).mean().iloc[-1] or close_spot <= pos['stop']):
                trigger_exit, reason = True, "SMA50 Trend Breach or Stop-Loss"
            elif sys['name'] == 'MODEL B (ZHAN)':
                ema20 = df['Close'].ewm(span=20).mean().iloc[-1]
                tr = pd.concat([df['High']-df['Low'], (df['High']-df['Close'].shift(1)).abs(), (df['Low']-df['Close'].shift(1)).abs()], axis=1).max(axis=1)
                if close_spot < (ema20 - 1.5 * tr.rolling(14).mean().iloc[-1]) or close_spot <= pos['stop']:
                    trigger_exit, reason = True, "Keltner Lower Boundary or Stop-Loss"
            elif sys['name'] == 'MODEL C (YIN)' and (pos['days_held'] >= 4 or close_spot > df['Close'].rolling(5).mean().iloc[-1] or close_spot <= pos['stop']):
                trigger_exit, reason = True, "Time-Stop/Mean Reversion/Stop-Loss"

            if trigger_exit:
                print(f"    🚨 [LIQUIDATE] Ticker: {t} | Reason: {reason} | ROUTE: MOC")
                has_exits = True
        
        # --- ENTRY LOGIC ---
        if market_ok and available_slots > 0:
            for t in TICKERS:
                if t in sys['positions'] or sector_cap_reached(sys['positions'], SECTORS[t]): continue
                
                df = data[t]
                close_spot, vol_spot = df['Close'].iloc[-1], df['Volume'].iloc[-1]
                sma50, ema20, atr14 = df['Close'].rolling(50).mean().iloc[-1], df['Close'].ewm(span=20).mean().iloc[-1], 1.0 # Simplified for brevity
                
                # Entry Logic Triggers (Insert your specific hole-fix logic blocks here)
                trigger_entry, mod_type, stop_price = False, "", 0.0
                
                # (Example Logic)
                if close_spot > sma50:
                    trigger_entry, mod_type, stop_price = True, "Momentum Breakout", close_spot * 0.95
                
                if trigger_entry:
                    friction_entry = apply_entry_friction(close_spot)
                    final_shares = int(min((total_equity * RISK_PER_TRADE) / (friction_entry - stop_price), (total_equity * MAX_ALLOC_FRACTION) / friction_entry))
                    if final_shares > 0:
                        print(f"    ⭐ [ENTRY] Ticker: {t} | Setup: {mod_type} | Shares: {final_shares} | ROUTE: MOC")
                        available_slots -= 1

if __name__ == "__main__":
    check_live_signals()
