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
    spy_sma50 = data['^GSPC']['Close'].rolling(50).mean().iloc[-1]

    # STRONGER REGIME FILTER
    market_ok = (spy_close > spy_sma200) and (spy_close > spy_sma50)
    
    print(f"S&P 500 Index Close: {spy_close:.2f} | 200 SMA: {spy_sma200:.2f} | 50 SMA: {spy_sma50:.2f}")
    print(f"Primary Regime Filter: {'[PASS]' if market_ok else '[FAIL]'}\n")

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
        
        print(f"  Adjusted Cash: ${current_cash:.2f} | Net Equity: ${total_equity:.2f}")
        print(f"  Available Slots: {available_slots} / {MAX_TOTAL_POSITIONS}")

        # --- A. EXIT LOGIC ---
        has_exits = False
        for t, pos in sys['positions'].items():
            df = data[t]
            close_spot = df['Close'].iloc[-1]

            tr = pd.concat([
                df['High'] - df['Low'],
                (df['High'] - df['Close'].shift(1)).abs(),
                (df['Low'] - df['Close'].shift(1)).abs()
            ], axis=1).max(axis=1)
            atr14 = tr.rolling(14).mean().iloc[-1]

            trigger_exit = False
            reason = ""

            if sys['name'] == 'MODEL A (YANG)' and (close_spot < df['Close'].rolling(50).mean().iloc[-1] or close_spot <= pos['stop']):
                trigger_exit, reason = True, "SMA50 Breach / Stop-Loss"

            elif sys['name'] == 'MODEL B (ZHAN)':
                ema20 = df['Close'].ewm(span=20).mean().iloc[-1]
                if close_spot < (ema20 - 1.5 * atr14) or close_spot <= pos['stop']:
                    trigger_exit, reason = True, "Keltner Lower / Stop-Loss"

            elif sys['name'] == 'MODEL C (YIN)' and (
                pos['days_held'] >= 4 or 
                close_spot > df['Close'].rolling(5).mean().iloc[-1] or 
                close_spot <= pos['stop']
            ):
                trigger_exit, reason = True, "Time-Stop / SMA5 / Stop-Loss"

            if trigger_exit:
                print(f"    🚨 [LIQUIDATE] {t} | Reason: {reason} | ROUTE: MOC")
                has_exits = True

        if not has_exits:
            print("    No positions require liquidation.")

        # --- B. ENTRY LOGIC (RANKED + SECTOR SAFE) ---
        print("  Active Allocation Triggers:")
        if market_ok and available_slots > 0:
            candidates = []

            for t in TICKERS:
                if t in sys['positions'] or sector_cap_reached(sys['positions'], SECTORS[t]):
                    continue

                df = data[t]
                close_spot = df['Close'].iloc[-1]
                vol_spot = df['Volume'].iloc[-1]
                sma50 = df['Close'].rolling(50).mean().iloc[-1]
                sma20 = df['Close'].rolling(20).mean().iloc[-1]
                ema20 = df['Close'].ewm(span=20).mean().iloc[-1]
                sma200 = df['Close'].rolling(200).mean().iloc[-1]
                avgvol30 = df['Volume'].rolling(30).mean().iloc[-1]
                prior_20high = df['High'].shift(1).rolling(20).max().iloc[-1]

                tr = pd.concat([
                    df['High'] - df['Low'],
                    (df['High'] - df['Close'].shift(1)).abs(),
                    (df['Low'] - df['Close'].shift(1)).abs()
                ], axis=1).max(axis=1)
                atr14 = tr.rolling(14).mean().iloc[-1]

                trigger_entry = False
                mod_type = ""
                stop_price = 0.0
                strength = 0.0

                # --- MODEL A (YANG) ---
                if sys['name'] == 'MODEL A (YANG)':
                    if close_spot > sma50 and vol_spot > avgvol30 and close_spot > prior_20high:
                        trigger_entry = True
                        mod_type = "Yang A (High Vol Breakout)"
                        stop_price = close_spot - (2 * atr14)
                        strength = (close_spot - prior_20high) / atr14 if atr14 > 0 else 0

                    elif close_spot > sma50 and vol_spot < avgvol30 and close_spot <= (sma20 + 2 * atr14):
                        trigger_entry = True
                        mod_type = "Yang B (Low Vol Pullback)"
                        stop_price = close_spot - (2 * atr14)
                        strength = (sma50 - close_spot) / atr14 if atr14 > 0 else 0

                # --- MODEL B (ZHAN) ---
                elif sys['name'] == 'MODEL B (ZHAN)':
                    kc_upper = ema20 + 1.5 * atr14
                    cons = all(
                        (df['Close'].iloc[-i] < (df['Close'].ewm(span=20).mean().iloc[-i] + 1.5 * tr.rolling(14).mean().iloc[-i]) and
                         df['Close'].iloc[-i] > (df['Close'].ewm(span=20).mean().iloc[-i] - 1.5 * tr.rolling(14).mean().iloc[-i]))
                        for i in range(2, 5)
                    )

                    if close_spot > kc_upper and vol_spot > avgvol30 and cons:
                        trigger_entry = True
                        mod_type = "Zhan A (Keltner Breakout)"
                        stop_price = close_spot - (1.5 * atr14)
                        strength = (close_spot - kc_upper) / atr14 if atr14 > 0 else 0

                    elif close_spot > ema20 and vol_spot < avgvol30 and cons:
                        trigger_entry = True
                        mod_type = "Zhan B (EMA Breakout)"
                        stop_price = close_spot - (1.5 * atr14)
                        strength = (close_spot - ema20) / atr14 if atr14 > 0 else 0

                # --- MODEL C (YIN) ---
                elif sys['name'] == 'MODEL C (YIN)':
                    delta = df['Close'].diff()
                    gain = delta.clip(lower=0)
                    loss = -delta.clip(upper=0)
                    rsi2 = (100 - (100 / (1 + (gain.rolling(2).mean() / loss.rolling(2).mean().replace(0, np.nan))))).iloc[-1]

                    atr50 = tr.rolling(50).mean().iloc[-1]
                    if atr50 > 0 and atr14 > 2 * atr50:
                        trigger_entry = False
                    else:
                        if close_spot > sma200:
                            if rsi2 < 5:
                                trigger_entry = True
                                mod_type = "Yin B (RSI < 5)"
                                stop_price = close_spot - (1.0 * atr14)
                                strength = (10 - rsi2)

                            elif rsi2 < 10:
                                trigger_entry = True
                                mod_type = "Yin A (RSI < 10)"
                                stop_price = close_spot - (1.0 * atr14)
                                strength = (10 - rsi2)

                if trigger_entry:
                    candidates.append({
                        'ticker': t,
                        'mod_type': mod_type,
                        'stop_price': stop_price,
                        'close_spot': close_spot,
                        'atr14': atr14,
                        'strength': strength
                    })

            # --- RANK BY STRENGTH ---
            candidates.sort(key=lambda x: x['strength'], reverse=True)

            # --- EXECUTE ENTRIES WITH SECTOR CAP ENFORCED ---
            for c in candidates:
                if available_slots <= 0:
                    break

                t = c['ticker']
                sector = SECTORS[t]
                if sector_cap_reached(sys['positions'], sector):
                    continue

                close_spot = c['close_spot']
                stop_price = c['stop_price']
                mod_type = c['mod_type']

                friction_entry = apply_entry_friction(close_spot)
                risk_dist = friction_entry - stop_price
                if risk_dist <= 0:
                    continue

                risk_cap = (total_equity * RISK_PER_TRADE) - COMMISSION
                shares_risk = risk_cap / risk_dist

                max_alloc = (total_equity * MAX_ALLOC_FRACTION) - COMMISSION
                shares_alloc = max_alloc / friction_entry

                final_shares = int(min(shares_risk, shares_alloc))
                if final_shares > 0:
                    print(f"    ⭐ [ENTRY] {t} | Setup: {mod_type} | Volume: {final_shares} | Entry: ${friction_entry:.2f} | Stop: ${stop_price:.2f}")
                    available_slots -= 1

        else:
            if not market_ok:
                print("    Market filter blocking entries.")
            if available_slots <= 0:
                print("    Portfolio at max capacity.")

        print("------------------------------------------------------------------------------\n")

if __name__ == "__main__":
    check_live_signals()
