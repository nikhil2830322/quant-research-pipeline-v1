import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime

# ==============================================================================
# CONFIGURATION & LEDGER (UPDATE BALANCES HERE EVERY AFTERNOON AFTER 3:15 PM)
# ==============================================================================
# Note: When tracking open positions, always format them like this:
# 'TICKER': {'sector': 'Tech', 'shares': 120, 'stop': 115.40, 'days_held': 1}
# days_held is auto-incremented by 1 each run for any position already
# present here, so you only need to paste in the correct starting value
# the first time a position is added -- no need to bump it by hand daily.
YANG_CASH, YANG_POSITIONS = 100000.00, {}
ZHAN_CASH, ZHAN_POSITIONS = 100000.00, {}
YIN_CASH, YIN_POSITIONS   = 100000.00, {}

# INPUT THE TARGET EXCEL ROW NUMBER YOU ARE PASTING INTO TODAY
TARGET_EXCEL_ROW = 2  # Change this to 4 tomorrow, 5 the next day, etc.

START_CAPITAL = 100000.00
RISK_PER_TRADE = 0.01
MAX_SECTOR_POSITIONS = 2
MAX_TOTAL_POSITIONS = 6
MAX_ALLOC_FRACTION = 1/6
SLIPPAGE_ENTRY = 0.0005
SLIPPAGE_EXIT = 0.0005
COMMISSION = 1.0
RISK_FREE_ANNUAL = 0.045
DAILY_RF = RISK_FREE_ANNUAL / 252

# Paste this clean configuration block right over your old one
TICKERS = ['MSFT', 'NVDA', 'AMZN', 'TSLA', 'JPM', 'GS', 'UNH', 'LLY', 'XOM', 'CAT', 'COST', 'WMT']

SECTORS = {
    'MSFT': 'Tech', 'NVDA': 'Tech',
    'AMZN': 'Consumer', 'TSLA': 'Consumer',
    'JPM': 'Finance', 'GS': 'Finance',
    'UNH': 'Healthcare', 'LLY': 'Healthcare',
    'XOM': 'Energy', 'CAT': 'Industrials',
    'COST': 'Staples', 'WMT': 'Staples'
}


def apply_entry_friction(price): return price * (1 + SLIPPAGE_ENTRY)
def apply_exit_friction(price): return price * (1 - SLIPPAGE_EXIT)
def sector_cap_reached(open_positions, sector):
    count = sum(1 for p in open_positions.values() if p['sector'] == sector)
    return count >= MAX_SECTOR_POSITIONS

# ==============================================================================
# MAIN ENGINE EXECUTION
# ==============================================================================
def check_live_signals():
    print("==============================================================================")
    print(f"LIVE TOURNAMENT EXECUTION DESK | RUNTIME: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} CDT")
    print("==============================================================================\n")

    data = yf.download(TICKERS + ['^GSPC'], period="3y", progress=False)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.swaplevel(0, 1)

    spy_close = data['^GSPC']['Close'].iloc[-1]
    spy_sma200 = data['^GSPC']['Close'].rolling(200).mean().iloc[-1]
    spy_sma50 = data['^GSPC']['Close'].rolling(50).mean().iloc[-1]

    market_ok = (spy_close > spy_sma200) and (spy_close > spy_sma50)
    regime_string = "PASS" if market_ok else "FAIL"

    print(f"S&P 500 Index Close: {spy_close:.2f} | 200 SMA Baseline: {spy_sma200:.2f} | 50 SMA Baseline: {spy_sma50:.2f}")
    print(f"Primary Regime Filter: [{regime_string}]\n")

    today_str = datetime.now().strftime('%Y-%m-%d')
    tab1_formula = f'=IF(AND(B{TARGET_EXCEL_ROW}>C{TARGET_EXCEL_ROW}, B{TARGET_EXCEL_ROW}>D{TARGET_EXCEL_ROW}), "PASS", "FAIL")'
    tab1_log = f"{today_str}\t{spy_close:.2f}\t{spy_sma200:.2f}\t{spy_sma50:.2f}\t100\t{tab1_formula}"

    systems = [
        {'name': 'MODEL A (YANG)', 'cash': YANG_CASH, 'positions': YANG_POSITIONS, 'id': 'YANG'},
        {'name': 'MODEL B (ZHAN)', 'cash': ZHAN_CASH, 'positions': ZHAN_POSITIONS, 'id': 'ZHAN'},
        {'name': 'MODEL C (YIN)', 'cash': YIN_CASH, 'positions': YIN_POSITIONS, 'id': 'YIN'}
    ]

    excel_outputs = {}

    for sys in systems:
        print(f"--- ACTIVE EXITS & ORDERS FOR {sys['name']} ---")

        # Auto-increment days_held for every position already in the ledger.
        # Anything present here was pasted in from a prior run, so it has
        # aged by exactly one trading day as of today's run. New entries
        # opened later in this same run (Phase B) are untouched and start at 0.
        for pos in sys['positions'].values():
            pos['days_held'] = pos.get('days_held', 0) + 1

        # Calculate current open equity BEFORE daily entry loops
        open_val_initial = sum(pos['shares'] * data[t]['Close'].iloc[-1] for t, pos in sys['positions'].items())
        current_cash = sys['cash'] * (1 + DAILY_RF)
        total_equity_initial = current_cash + open_val_initial
        available_slots = MAX_TOTAL_POSITIONS - len(sys['positions'])

        print(f"  Adjusted Cash Account: ${current_cash:.2f} | Net Portfolio Equity: ${total_equity_initial:.2f}")
        print(f"  Available Portfolio Allocation Slots: {available_slots} / {MAX_TOTAL_POSITIONS}")

        # --- PHASE A: LIQUIDATION CHECK ---
        has_exits = False
        for t, pos in sys['positions'].items():
            df = data[t]
            close_spot = df['Close'].iloc[-1]
            trigger_exit, reason = False, ""
            tr = pd.concat([df['High']-df['Low'], (df['High']-df['Close'].shift(1)).abs(), (df['Low']-df['Close'].shift(1)).abs()], axis=1).max(axis=1)
            atr14 = tr.rolling(14).mean().iloc[-1]

            # Safe stop/days_held handling in case ledger entry was pasted without these fields
            current_stop = pos.get('stop', close_spot - (2.0 * atr14))
            days_held = pos.get('days_held', 0)

            if sys['name'] == 'MODEL A (YANG)' and (close_spot < df['Close'].rolling(50).mean().iloc[-1] or close_spot <= current_stop):
                trigger_exit, reason = True, "SMA50 Trend Breach or Stop-Loss"
            elif sys['name'] == 'MODEL B (ZHAN)':
                ema20 = df['Close'].ewm(span=20).mean().iloc[-1]
                if close_spot < (ema20 - 1.5 * atr14) or close_spot <= current_stop:
                    trigger_exit, reason = True, "Keltner Lower Boundary Breach"
            elif sys['name'] == 'MODEL C (YIN)' and (days_held >= 4 or close_spot > df['Close'].rolling(5).mean().iloc[-1] or close_spot <= current_stop):
                trigger_exit, reason = True, "Time-Stop / Mean Reversion"

            if trigger_exit:
                print(f"    \U0001f6a8 [LIQUIDATE] Ticker: {t} | Reason: {reason} | ROUTE: MOC")
                has_exits = True
        if not has_exits:
            print("    No open positions satisfy liquidation rules today.")

        # --- PHASE B: CANDIDATE SIGNAL GENERATION & RANKING ---
        print("  Active Allocation Triggers:")
        if market_ok and available_slots > 0:
            candidates = []
            for t in TICKERS:
                if t in sys['positions'] or sector_cap_reached(sys['positions'], SECTORS[t]): continue
                df = data[t]
                close_spot, vol_spot = df['Close'].iloc[-1], df['Volume'].iloc[-1]
                sma50, sma20, ema20 = df['Close'].rolling(50).mean().iloc[-1], df['Close'].rolling(20).mean().iloc[-1], df['Close'].ewm(span=20).mean().iloc[-1]
                avgvol30 = df['Volume'].rolling(30).mean().iloc[-1]
                prior_20high = df['High'].shift(1).rolling(20).max().iloc[-1]
                tr = pd.concat([df['High']-df['Low'], (df['High']-df['Close'].shift(1)).abs(), (df['Low']-df['Close'].shift(1)).abs()], axis=1).max(axis=1)
                atr14 = tr.rolling(14).mean().iloc[-1]

                trigger_entry, mod_type, stop_price, strength = False, "", 0.0, 0.0

                if sys['name'] == 'MODEL A (YANG)':
                    if close_spot > sma50 and vol_spot > avgvol30 and close_spot > prior_20high:
                        trigger_entry, mod_type = True, "Yang Module A (High Vol Breakout)"
                        stop_price = close_spot - (2 * atr14)
                        strength = (close_spot - prior_20high) / atr14 if atr14 > 0 else 0
                    elif close_spot > sma50 and vol_spot < avgvol30 and close_spot <= (sma20 + 2 * atr14):
                        trigger_entry, mod_type = True, "Yang Module B (Low Vol Pullback)"
                        stop_price = close_spot - (2 * atr14)
                        strength = (sma50 - close_spot) / atr14 if atr14 > 0 else 0

                elif sys['name'] == 'MODEL B (ZHAN)':
                    kc_upper = ema20 + 1.5 * atr14
                    # Consolidation check includes today's close (i=1) through two days prior (i=3)
                    cons = all((df['Close'].iloc[-i] < (df['Close'].ewm(span=20).mean().iloc[-i] + 1.5 * tr.rolling(14).mean().iloc[-i]) and \
                                df['Close'].iloc[-i] > (df['Close'].ewm(span=20).mean().iloc[-i] - 1.5 * tr.rolling(14).mean().iloc[-i])) for i in range(1,4))
                    if close_spot > kc_upper and vol_spot > avgvol30 and cons:
                        trigger_entry, mod_type = True, "Zhan Module A (Keltner Breakout)"
                        stop_price = close_spot - (1.5 * atr14)
                        strength = (close_spot - kc_upper) / atr14 if atr14 > 0 else 0
                    elif close_spot > ema20 and vol_spot < avgvol30 and cons:
                        trigger_entry, mod_type = True, "Zhan Module B (EMA Breakout)"
                        stop_price = close_spot - (1.5 * atr14)
                        strength = (close_spot - ema20) / atr14 if atr14 > 0 else 0

                elif sys['name'] == 'MODEL C (YIN)':
                    delta = df['Close'].diff()
                    gain, loss = delta.clip(lower=0), -delta.clip(upper=0)
                    rsi2 = (100 - (100 / (1 + (gain.rolling(2).mean() / loss.rolling(2).mean().replace(0, np.nan))))).iloc[-1]
                    atr50 = tr.rolling(50).mean().iloc[-1]

                    if not (atr50 > 0 and atr14 > 2 * atr50) and close_spot > df['Close'].rolling(200).mean().iloc[-1]:
                        # Mutually exclusive tiers so a candidate is only ever scored once,
                        # with deep-exhaustion setups ranked above standard pullbacks
                        if rsi2 < 5:
                            trigger_entry, mod_type = True, "Yin Module B (Deep Exhaustion Pullback)"
                            stop_price = close_spot - (1.0 * atr14)
                            strength = 15 - rsi2
                        elif rsi2 < 10:
                            trigger_entry, mod_type = True, "Yin Module A (Standard Pullback)"
                            stop_price = close_spot - (1.0 * atr14)
                            strength = 10 - rsi2

                if trigger_entry:
                    candidates.append({'ticker': t, 'mod_type': mod_type, 'stop_price': stop_price, 'close_spot': close_spot, 'strength': strength})

            candidates.sort(key=lambda x: x['strength'], reverse=True)

            for c in candidates:
                if available_slots <= 0 or sector_cap_reached(sys['positions'], SECTORS[c['ticker']]): continue
                friction_entry = apply_entry_friction(c['close_spot'])
                risk_dist = friction_entry - c['stop_price']
                if risk_dist > 0:
                    risk_cap = (total_equity_initial * RISK_PER_TRADE) - COMMISSION
                    max_alloc = (total_equity_initial * MAX_ALLOC_FRACTION) - COMMISSION
                    final_shares = int(min(risk_cap / risk_dist, max_alloc / friction_entry))
                    if final_shares > 0:
                        print(f"    \u2b50 [ENTRY] Ticker: {c['ticker']} | Volume: {final_shares} Shares | Entry Price: ${friction_entry:.2f} | Setup: {c['mod_type']}")
                        # Store stop and days_held directly on the position so tomorrow's
                        # exit check doesn't have to fall back to defaults
                        sys['positions'][c['ticker']] = {
                            'sector': SECTORS[c['ticker']],
                            'shares': final_shares,
                            'stop': c['stop_price'],
                            'days_held': 0
                        }
                        current_cash -= ((final_shares * friction_entry) + COMMISSION)
                        available_slots -= 1

        # --- POST-ENTRY LEDGER COMPILATION ---
        open_val_final = sum(pos['shares'] * data[t]['Close'].iloc[-1] for t, pos in sys['positions'].items())
        ticker_list_str = ", ".join(sys['positions'].keys()) if sys['positions'] else "None"
        sector_counts = {}
        for p in sys['positions'].values():
            sector_counts[p['sector']] = sector_counts.get(p['sector'], 0) + 1
        sector_str = ", ".join([f"{k}:{v}" for k, v in sector_counts.items()]) if sector_counts else "None"

        equity_formula = f"=C{TARGET_EXCEL_ROW}+H{TARGET_EXCEL_ROW}"
        return_formula = f'=IF(B{TARGET_EXCEL_ROW-1}=0, 0, (B{TARGET_EXCEL_ROW}-B{TARGET_EXCEL_ROW-1})/B{TARGET_EXCEL_ROW-1})'

        sys_row = f"{today_str}\t{equity_formula}\t{current_cash:.2f}\t{return_formula}\t{len(sys['positions'])}\t{ticker_list_str}\t{sector_str}\t{open_val_final:.2f}"
        excel_outputs[sys['id']] = sys_row
        print("------------------------------------------------------------------------------\n")

    # ==============================================================================
    # \U0001f4cb EXCEL ONE-CLICK COPY DASHBOARD
    # ==============================================================================
    print("==============================================================================")
    print("          \U0001f4cb EXCEL ONE-CLICK COPY DASHBOARD (TAB-DELIMITED FORMAT)")
    print("==============================================================================")
    print(f"\U0001f449 FOR TAB 1 (Global Market Filter) -> Paste into Cell A{TARGET_EXCEL_ROW}:\n{tab1_log}\n")
    print(f"\U0001f449 FOR TAB 2 (Yang Portfolio Row)  -> Paste into Cell A{TARGET_EXCEL_ROW}:\n{excel_outputs['YANG']}\n")
    print(f"\U0001f449 FOR TAB 3 (Zhan Portfolio Row)  -> Paste into Cell A{TARGET_EXCEL_ROW}:\n{excel_outputs['ZHAN']}\n")
    print(f"\U0001f449 FOR TAB 4 (Yin Portfolio Row)   -> Paste into Cell A{TARGET_EXCEL_ROW}:\n{excel_outputs['YIN']}\n")
    print("==============================================================================")

if __name__ == "__main__":
    check_live_signals()
