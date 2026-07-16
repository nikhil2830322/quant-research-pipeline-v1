import numpy as np
import pandas as pd
from config_and_data import SystemContext, sectors
from risk_and_friction import (
    apply_cash_drag, apply_exit_friction, apply_entry_friction,
    gap_down_exit, position_size, sector_cap_reached
)

# ==============================================================================
# MODEL B — ZHAN (Volatility Expansion, with trailing stops) - POINT-IN-TIME OPTIMIZED [PART 1]
# ==============================================================================
def run_zhan(ind: dict, data: dict, sp500: pd.Series,
             sp500_sma200: pd.Series, sp500_sma50: pd.Series,
             ctx: SystemContext):

    cash = ctx.START_CAPITAL
    equity = ctx.START_CAPITAL
    open_positions = {}
    trade_log = []
    equity_curve = []
    daily_returns = []

    dates = sp500.index[200:]

    # Precompute market regime filter
    market_ok_series = (sp500 > sp500_sma200) & (sp500 > sp500_sma50)

    # Precompute sector map (Dynamic PIT Refactor)
    ticker_to_sector = {t: sectors.get(t, "Unassigned") for t in ind.keys()}

    # Cache indicator arrays for speed (Dynamic PIT Refactor)
    cached_ind = {}
    for t in ind.keys(): # <-- Swapped 'tickers' to 'ind.keys()' to isolate active historical assets
        df = ind.get(t)
        if df is None:
            continue

        idx_to_pos = {date_val: pos for pos, date_val in enumerate(df.index)}

        cached_ind[t] = {
            'Close': df['Close'].values,
            'KC_LOWER': df['KC_LOWER'].values,
            'KC_UPPER': df['KC_UPPER'].values,
            'EMA20': df['EMA20'].values,
            'Vol': df['Vol'].values,
            'AvgVol30': df['AvgVol30'].values,
            'ATR14': df['ATR14'].values,
            '_idx_to_pos': idx_to_pos,
            'raw_close_series': df['Close']
        }

    # ==============================================================================
    # MAIN LOOP
    # ==============================================================================
    for i in range(1, len(dates)):
        date = dates[i]
        prev_date = dates[i - 1]

        cash = apply_cash_drag(cash, ctx)
        market_ok = market_ok_series.get(prev_date, False)

        just_exited_today = set()
        to_close = []

        # ----------------------------------------------------------------------
        # 1. EXIT ENGINE — trailing stops + KC breakdown
        # ----------------------------------------------------------------------
        for t, pos in open_positions.items():
            t_ind = cached_ind.get(t)
            if t_ind is None:
                continue

            curr_idx = t_ind['_idx_to_pos'].get(prev_date)
            if curr_idx is None:
                continue

            prev_close = t_ind['Close'][curr_idx]
            prev_kc_lower = t_ind['KC_LOWER'][curr_idx]
            y_atr = t_ind['ATR14'][curr_idx]

            # Trailing stop update
            if not np.isnan(prev_close) and not np.isnan(y_atr) and y_atr > 0:
                new_stop = prev_close - (pos['stop_mult'] * y_atr)
                if new_stop > pos['stop']:
                    pos['stop'] = new_stop

            # Exit conditions
            if prev_close < prev_kc_lower or prev_close <= pos['stop']:
                to_close.append(t)

        # Execute exits
        for t in to_close:
            pos = open_positions.pop(t)
            just_exited_today.add(t)

            if t not in data['Open'] or date not in data['Open'][t].index:
                continue

            open_price = data['Open'][t].loc[date]
            t_ind = cached_ind.get(t)

            idx_pos = t_ind['_idx_to_pos'].get(date)
            close_price = t_ind['Close'][idx_pos] if idx_pos is not None else open_price

            exit_price = gap_down_exit(open_price, pos['stop'], close_price)
            exit_price = apply_exit_friction(exit_price, ctx)
            exit_price = max(exit_price, 0.0001)

            cash += pos['shares'] * exit_price
            cash -= ctx.COMMISSION

            pnl = (exit_price - pos['entry']) * pos['shares'] - ctx.COMMISSION

            trade_log.append({
                'ticker': t,
                'entry_date': pos['entry_date'],
                'exit_date': date,
                'entry_price': pos['entry'],
                'exit_price': exit_price,
                'shares': pos['shares'],
                'pnl': pnl,
                'mod_type': pos.get('mod_type', 'Unknown')
            })

         # ----------------------------------------------------------------------
        # 2. ENTRY ENGINE — volatility expansion
        # ----------------------------------------------------------------------
        if market_ok:
            available_slots = ctx.MAX_TOTAL_POSITIONS - len(open_positions)
            candidates = []

            for t in ind.keys():  # <-- Swapped 'tickers' to 'ind.keys()' to prevent structural crashes
                if t in open_positions or t in just_exited_today:
                    continue

                t_ind = cached_ind.get(t)
                if t_ind is None:
                    continue

                curr_idx = t_ind['_idx_to_pos'].get(prev_date)
                if curr_idx is None or curr_idx < 4:
                    continue

                y_close = t_ind['Close'][curr_idx]
                y_ema20 = t_ind['EMA20'][curr_idx]
                y_kc_upper = t_ind['KC_UPPER'][curr_idx]
                y_vol = t_ind['Vol'][curr_idx]
                y_avgvol30 = t_ind['AvgVol30'][curr_idx]
                y_atr = t_ind['ATR14'][curr_idx]

                # NaN guard
                if (
                    np.isnan(y_close) or np.isnan(y_ema20) or np.isnan(y_kc_upper) or
                    np.isnan(y_vol) or np.isnan(y_avgvol30) or np.isnan(y_atr) or y_atr <= 0
                ):
                    continue

                c_close = t_ind['Close']
                c_upper = t_ind['KC_UPPER']
                c_lower = t_ind['KC_LOWER']

                # Consolidation check
                if (
                    np.isnan(c_close[curr_idx - 1]) or np.isnan(c_close[curr_idx - 2]) or np.isnan(c_close[curr_idx - 3]) or
                    np.isnan(c_upper[curr_idx - 1]) or np.isnan(c_upper[curr_idx - 2]) or np.isnan(c_upper[curr_idx - 3]) or
                    np.isnan(c_lower[curr_idx - 1]) or np.isnan(c_lower[curr_idx - 2]) or np.isnan(c_lower[curr_idx - 3])
                ):
                    continue

                cons = (
                    c_close[curr_idx - 1] < c_upper[curr_idx - 1] and c_close[curr_idx - 1] > c_lower[curr_idx - 1] and
                    c_close[curr_idx - 2] < c_upper[curr_idx - 2] and c_close[curr_idx - 2] > c_lower[curr_idx - 2] and
                    c_close[curr_idx - 3] < c_upper[curr_idx - 3] and c_close[curr_idx - 3] > c_lower[curr_idx - 3]
                )

                trigger_entry = False
                mod_type = ""
                strength = 0.0
                stop_mult = 1.5

                # Module A — Keltner breakout
                if y_close > y_kc_upper and y_vol > y_avgvol30 and cons:
                    trigger_entry = True
                    mod_type = "Zhan Module A (Keltner Breakout)"
                    strength = (y_close - y_kc_upper) / y_atr

                # Module B — EMA breakout
                elif y_close > y_ema20 and y_vol < y_avgvol30 and cons:
                    trigger_entry = True
                    mod_type = "Zhan Module B (EMA Breakout)"
                    strength = (y_close - y_ema20) / y_atr

                if trigger_entry:
                    candidates.append({
                        'ticker': t,
                        'mod_type': mod_type,
                        'atr': y_atr,
                        'strength': strength,
                        'sector': ticker_to_sector.get(t, "Unassigned"),  # Robust dictionary safety hook
                        'stop_mult': stop_mult
                    })

            # Sort by strength
            candidates.sort(key=lambda x: x['strength'], reverse=True)

            sizing_equity = equity

            # Allocate
            for c in candidates:
                if available_slots <= 0:
                    break

                t = c['ticker']
                sector = c['sector']

                if sector_cap_reached(open_positions, sector, ctx):
                    continue

                if t not in data['Open'] or date not in data['Open'][t].index:
                    continue

                raw_open = data['Open'][t].loc[date]
                if pd.isna(raw_open) or raw_open <= 0:
                    continue

                entry_price = apply_entry_friction(raw_open, ctx)
                stop = entry_price - (c['stop_mult'] * c['atr'])

                shares = position_size(entry_price, stop, sizing_equity, ctx)
                if shares <= 0:
                    continue

                total_cost = (shares * entry_price) + ctx.COMMISSION

                if cash >= total_cost:
                    cash -= total_cost
                    open_positions[t] = {
                        'entry': entry_price,
                        'stop': stop,
                        'shares': shares,
                        'sector': sector,
                        'entry_date': date,
                        'stop_mult': c['stop_mult'],
                        'days_held': 0,
                        'mod_type': c['mod_type']
                    }
                    available_slots -= 1
                    sizing_equity -= total_cost

        # ----------------------------------------------------------------------
        # 3. DAILY VALUATION
        # ----------------------------------------------------------------------
        open_val = 0.0
        for t, pos in open_positions.items():
            t_ind = cached_ind.get(t)
            if t_ind is not None:
                valuation_price = t_ind['raw_close_series'].asof(date)
                if pd.isna(valuation_price):
                    valuation_price = pos['entry']
                open_val += pos['shares'] * valuation_price
            else:
                open_val += pos['shares'] * pos['entry']

        prev_equity = equity
        equity = cash + open_val

        # FIXED: Log returns for every single tracked trading day to preserve matrix dimension shapes
        daily_returns.append((equity - prev_equity) / prev_equity)
        equity_curve.append(equity)

    return {
        'name': 'Zhan',
        'equity_curve': equity_curve,
        'trade_log': trade_log,
        'daily_returns': daily_returns,
        'dates': list(dates[1:])  # Aligns flawlessly with daily returns bounds
    }

