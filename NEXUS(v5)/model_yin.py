import numpy as np
import pandas as pd
from config_and_data import SystemContext, sectors
from risk_and_friction import (
    apply_cash_drag, apply_exit_friction, apply_entry_friction,
    gap_down_exit, position_size, sector_cap_reached
)

# ==============================================================================
# MODEL C - YIN (Mean Reversion) - POINT-IN-TIME OPTIMIZED [PART 1]
# ==============================================================================
def run_yin(ind: dict, data: dict, sp500: pd.Series, sp500_sma200: pd.Series, sp500_sma50: pd.Series, ctx: SystemContext):
    cash = ctx.START_CAPITAL
    equity = ctx.START_CAPITAL
    open_positions = {}
    trade_log = []
    equity_curve = []
    daily_returns = []

    # Fast-path optimization: Pre-align and cache series to prevent daily index-lookup overhead
    dates = sp500.index[200:]
    
    # Pre-calculate market regime condition vectorially for speed
    market_ok_series = (sp500 > sp500_sma200) & (sp500 > sp500_sma50)
    
    # Cache dictionary structures of your indicators for faster key access
    # Maps ticker -> column -> Series/Dict for fast scalar access
    cached_ind = {}
    for t in ind.keys(): # <-- Swapped 'tickers' to 'ind.keys()' to capture dynamic eras safely
        df = ind.get(t)
        if df is not None:
            cached_ind[t] = {col: df[col] for col in df.columns if col in ['Close', 'SMA5', 'RSI2', 'SMA200', 'ATR14', 'ATR50']}

    for i in range(1, len(dates)):
        date = dates[i]
        prev_date = dates[i-1]
        
        # Apply structural cash drag
        cash = apply_cash_drag(cash, ctx)
        
        # Market Regime Filter
        market_ok = market_ok_series.get(prev_date, False)
        just_exited_today = set()
        to_close = []

        # ----------------------------------------------------------------------
        # 1. EVALUATE EXITS
        # ----------------------------------------------------------------------
        for t, pos in open_positions.items():
            t_ind = cached_ind.get(t)
            if t_ind is None or prev_date not in t_ind['Close'].index:
                continue
                
            prev_close = t_ind['Close'].loc[prev_date]
            prev_sma5 = t_ind['SMA5'].loc[prev_date]

            # Time-based stop (4 days) or Mean Reversion Target reached (Close > SMA5) or Stop Loss Hit
            if pos['days_held'] >= 4:
                to_close.append(t)
            elif prev_close > prev_sma5 or prev_close <= pos['stop']:
                to_close.append(t)

        # Process exits using current day opening action
        for t in to_close:
            pos = open_positions.pop(t)
            just_exited_today.add(t)
            
            if t not in data['Open'] or date not in data['Open'][t].index:
                continue
                
            open_price = data['Open'][t].loc[date]
            
            t_ind = cached_ind.get(t)
            close_price = t_ind['Close'].loc[date] if (t_ind and date in t_ind['Close'].index) else open_price
            
            # Formulate exit price accounting for overnight gap risk and slippage
            exit_price = gap_down_exit(open_price, pos['stop'], close_price)
            exit_price = apply_exit_friction(exit_price, ctx)
            
            # Prevent impossible negative pricing artifacts from friction math
            exit_price = max(exit_price, 0.0001)

            cash += pos['shares'] * exit_price
            cash -= ctx.COMMISSION

            pnl = (exit_price - pos['entry']) * pos['shares'] - ctx.COMMISSION
            trade_log.append({
                'ticker': t, 'entry_date': pos['entry_date'], 'exit_date': date,
                'entry_price': pos['entry'], 'exit_price': exit_price,
                'shares': pos['shares'], 'pnl': pnl
            })

        # ----------------------------------------------------------------------
        # 2. EVALUATE ENTRIES
        # ----------------------------------------------------------------------
        if market_ok:
            available_slots = ctx.MAX_TOTAL_POSITIONS - len(open_positions)
            candidates = []

            for t in ind.keys(): # <-- Swapped 'tickers' to 'ind.keys()' to prevent data-gap crashes
                if t in open_positions or t in just_exited_today:
                    continue
                    
                t_ind = cached_ind.get(t)
                if t_ind is None or prev_date not in t_ind['Close'].index:
                    continue

                y_close = t_ind['Close'].loc[prev_date]
                y_rsi2 = t_ind['RSI2'].loc[prev_date]
                y_sma200 = t_ind['SMA200'].loc[prev_date]
                y_atr14 = t_ind['ATR14'].loc[prev_date]
                y_whites = t_ind['ATR50'].loc[prev_date] if 'ATR50' in t_ind else None # Safe tracking gate

                # Defensive checks for structural completeness
                if pd.isna(y_close) or pd.isna(y_rsi2) or pd.isna(y_sma200) or pd.isna(y_atr14):
                    continue

                sector = sectors.get(t, "Unassigned") # Refactored crash fallback
                if sector_cap_reached(open_positions, sector, ctx):
                    continue
                    
                # Volatility explosion filter: Avoid catching falling knives when short-term vol > 2x structural vol
                if y_whites is not None and pd.notna(y_whites) and y_whites > 0 and y_atr14 > 2 * y_whites:
                    continue

                trigger_entry = False
                mod_type = ""
                strength = 0.0

                # Mean reversion prioritizing mechanism based on extreme oversold levels
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

            # Sort by highest oversold strength score 
            candidates.sort(key=lambda x: x['strength'], reverse=True)
            
            # Set target sizing equity using the static starting daily equity snapshot
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
                if pd.isna(raw_open) or raw_open <= 0:
                    continue
                    
                stop = raw_open - (1.0 * c['atr'])
                
                # Calculate size using current portfolio dynamic capital value
                shares = position_size(raw_open, stop, sizing_equity, ctx)
                if shares <= 0:
                    continue
                    
                entry_price = apply_entry_friction(raw_open, ctx)
                total_cost = (shares * entry_price) + ctx.COMMISSION

                if cash >= total_cost:
                    cash -= total_cost
                    open_positions[t] = {
                        'entry': entry_price, 'stop': stop, 'shares': shares,
                        'sector': sector, 'entry_date': date, 'days_held': 0
                    }
                    available_slots -= 1
                    # Sizing equity tracks exact available capital context dynamically
                    sizing_equity -= total_cost 

        # Increment age tracking across open trades
        for t in open_positions:
            open_positions[t]['days_held'] += 1

        # ----------------------------------------------------------------------
        # 3. DAILY PORTFOLIO VALUATION
        # ----------------------------------------------------------------------
        open_val = 0.0
        for t, pos in open_positions.items():
            t_ind = cached_ind.get(t)
            if t_ind is not None:
                # Realism Fix: Forward-fill values if an asset has an active gap or trading halt
                idx = t_ind['Close'].index
                available_dates = idx[idx <= date]
                if not available_dates.empty:
                    last_valid_date = available_dates[-1]
                    open_val += pos['shares'] * t_ind['Close'].loc[last_valid_date]
                else:
                    open_val += pos['shares'] * pos['entry']
            else:
                open_val += pos['shares'] * pos['entry']
	
        prev_equity = equity
        equity = cash + open_val
        
        # FIXED: Log returns for every single tracked trading day to prevent array mismatch errors down the line
        daily_returns.append((equity - prev_equity) / prev_equity)
        equity_curve.append(equity)

    return {
        'name': 'Yin', 
        'equity_curve': equity_curve, 
        'trade_log': trade_log,
        'daily_returns': daily_returns, 
        'dates': list(dates[1:])  # Aligns flawlessly with daily_returns matrix bounds
    }
