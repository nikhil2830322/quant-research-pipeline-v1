import numpy as np
import pandas as pd
from config_and_data import SystemContext, tickers, sectors
from risk_and_friction import (
    apply_cash_drag, apply_exit_friction, apply_entry_friction,
    gap_down_exit, position_size, sector_cap_reached
)

# ==============================================================================
# MODEL A - YANG (Trend Following, with trailing stops) - OPTIMIZED
# ==============================================================================
def run_yang(ind: dict, data: dict, sp500: pd.Series, sp500_sma200: pd.Series, sp500_sma50: pd.Series, ctx: SystemContext):
    cash = ctx.START_CAPITAL
    equity = ctx.START_CAPITAL
    open_positions = {}
    trade_log = []
    equity_curve = []
    daily_returns = []

    dates = sp500.index[200:]
    
    # Pre-calculate market regime condition vectorially for speed
    market_ok_series = (sp500 > sp500_sma200) & (sp500 > sp500_sma50)
    
    # High-Performance Memory Cache Map
    # Extract targeted columns to bypass multi-level hash-tree searches on DataFrames
    cached_ind = {}
    required_cols = {'Close', 'SMA50', 'ATR14', 'Prior20High', 'Vol', 'AvgVol30', 'SMA20'}
    for t in ind.keys(): # <-- Swapped 'tickers' to 'ind.keys()'
        df = ind.get(t)
        if df is not None:
            cached_ind[t] = {col: df[col] for col in df.columns if col in required_cols}

    for i in range(1, len(dates)):
        date = dates[i]
        prev_date = dates[i-1]
        cash = apply_cash_drag(cash, ctx)

        market_ok = market_ok_series.get(prev_date, False)
        just_exited_today = set()
        to_close = []

        # ----------------------------------------------------------------------
        # 1. EVALUATE TRAILING STOPS & EXITS
        # ----------------------------------------------------------------------
        for t, pos in open_positions.items():
            t_ind = cached_ind.get(t)
            if t_ind is None or prev_date not in t_ind['Close'].index:
                continue
                
            prev_close = t_ind['Close'].loc[prev_date]
            prev_sma50 = t_ind['SMA50'].loc[prev_date]
            y_atr = t_ind['ATR14'].loc[prev_date]

            # Update Trailing Stop Mechanics cleanly if data is fully formed
            if pd.notna(prev_close) and pd.notna(y_atr):
                calculated_trailing_stop = prev_close - (pos['stop_mult'] * y_atr)
                if calculated_trailing_stop > pos['stop']:
                    pos['stop'] = calculated_trailing_stop

            # Structural Trend Break or Trailing Stop Breach
            if prev_close < prev_sma50 or prev_close <= pos['stop']:
                to_close.append(t)

        # Process liquidations using market opening execution rules
        for t in to_close:
            pos = open_positions.pop(t)
            just_exited_today.add(t)
            
            if t not in data['Open'] or date not in data['Open'][t].index:
                continue
                
            open_price = data['Open'][t].loc[date]
            t_ind = cached_ind.get(t)
            close_price = t_ind['Close'].loc[date] if (t_ind and date in t_ind['Close'].index) else open_price
            
            exit_price = gap_down_exit(open_price, pos['stop'], close_price)
            exit_price = apply_exit_friction(exit_price, ctx)
            exit_price = max(exit_price, 0.0001)  # Safeguard negative balance artifacts

            cash += pos['shares'] * exit_price
            cash -= ctx.COMMISSION

            pnl = (exit_price - pos['entry']) * pos['shares'] - ctx.COMMISSION
            trade_log.append({
                'ticker': t, 'entry_date': pos['entry_date'], 'exit_date': date,
                'entry_price': pos['entry'], 'exit_price': exit_price,
                'shares': pos['shares'], 'pnl': pnl
            })

        # ----------------------------------------------------------------------
        # 2. EVALUATE ENTRIES (TREND BREAKOUTS / PULLBACKS)
        # ----------------------------------------------------------------------
        if market_ok:
            available_slots = ctx.MAX_TOTAL_POSITIONS - len(open_positions)
            candidates = []

            for t in ind.keys():  # <-- Swapped 'tickers' to 'ind.keys()'
                if t in open_positions or t in just_exited_today:
                    continue
                    
                t_ind = cached_ind.get(t)
                if t_ind is None or prev_date not in t_ind['Close'].index:
                    continue

                y_close = t_ind['Close'].loc[prev_date]
                y_sma50 = t_ind['SMA50'].loc[prev_date]
                y_prior_high = t_ind['Prior20High'].loc[prev_date]
                y_vol = t_ind['Vol'].loc[prev_date]
                y_avgvol30 = t_ind['AvgVol30'].loc[prev_date]
                y_atr = t_ind['ATR14'].loc[prev_date]
                y_sma20 = t_ind['SMA20'].loc[prev_date]

                # Defensive missing value gate
                if pd.isna(y_close) or pd.isna(y_sma50) or pd.isna(y_atr):
                    continue

                sector = sectors.get(t, "Unassigned")
                if sector_cap_reached(open_positions, sector, ctx):
                    continue

                trigger_entry = False
                mod_type = ""
                stop_mult = 2.0
                strength = 0.0

                # Module A: High Volume Momentum Breakout 
                if y_close > y_sma50 and pd.notna(y_vol) and pd.notna(y_avgvol30) and y_vol > y_avgvol30 and pd.notna(y_prior_high) and y_close > y_prior_high:
                    trigger_entry = True
                    mod_type = "Yang Module A (High Vol Breakout)"
                    strength = (y_close - y_prior_high) / y_atr if y_atr > 0 else 0.0
                    
                # Module B: Low Volume Controlled Mean Pullback
                elif y_close > y_sma50 and pd.notna(y_vol) and pd.notna(y_avgvol30) and y_vol < y_avgvol30 and pd.notna(y_sma20) and y_close <= (y_sma20 + 2 * y_atr):
                    trigger_entry = True
                    mod_type = "Yang Module B (Low Vol Pullback)"
                    strength = (y_sma50 - y_close) / y_atr if y_atr > 0 else 0.0

                if trigger_entry:
                    candidates.append({
                        'ticker': t, 'mod_type': mod_type, 'stop_mult': stop_mult,
                        'atr': y_atr, 'strength': strength, 'sector': sector
                    })

            # Rank by structural signal intensity
            candidates.sort(key=lambda x: x['strength'], reverse=True)
            
            # Establish static baseline pool mapping for unified entry sizing
            sizing_equity = equity

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
                if pd.isna(raw_open) or raw_open <= 0 or pd.isna(c['atr']) or c['atr'] <= 0:
                    continue
                    
                stop = raw_open - (c['stop_mult'] * c['atr'])
                
                # Size position leveraging unified portfolio equity snapshot
                shares = position_size(raw_open, stop, sizing_equity, ctx)
                if shares <= 0:
                    continue
                    
                entry_price = apply_entry_friction(raw_open, ctx)
                total_cost = (shares * entry_price) + ctx.COMMISSION

                if cash >= total_cost:
                    cash -= total_cost
                    open_positions[t] = {
                        'entry': entry_price, 'stop': stop, 'shares': shares,
                        'sector': sector, 'entry_date': date,
                        'stop_mult': c['stop_mult'], 'days_held': 0
                    }
                    available_slots -= 1
                    # Account for dynamic cash allocation strictly
                    sizing_equity -= total_cost

        # Progress holding metadata across active executions
        for t in open_positions:
            open_positions[t]['days_held'] = open_positions[t].get('days_held', 0) + 1

        # ----------------------------------------------------------------------
        # 3. DAILY PORTFOLIO VALUATION
        # ----------------------------------------------------------------------
        open_val = 0.0
        for t, pos in open_positions.items():
            t_ind = cached_ind.get(t)
            if t_ind is not None:
                idx = t_ind['Close'].index
                available_dates = idx[idx <= date]
                if not available_dates.empty:
                    last_valid_date = available_dates[-1]
                    open_val += pos['shares'] * t_ind['Close'].loc[last_valid_date]
                else:
                    open_val += pos['shares'] * pos['entry']
            else:
                open_val += pos['shares'] * pos['entry']

               # Clean corporate valuation alignment
        prev_equity = equity
        equity = cash + open_val
        
        # FIXED: Log returns for every single tracked trading day
        daily_returns.append((equity - prev_equity) / prev_equity)
        equity_curve.append(equity)

    return {
        'name': 'Yang', 
        'equity_curve': equity_curve, 
        'trade_log': trade_log,
        'daily_returns': daily_returns, 
        'dates': list(dates[1:])  # This matches the daily returns perfectly
    }

#6k characters to 9k lol