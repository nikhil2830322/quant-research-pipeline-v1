import numpy as np
import pandas as pd
from config_and_data import SystemContext, sectors
from risk_and_friction import (
    apply_cash_drag, apply_exit_friction, apply_entry_friction,
    gap_down_exit, position_size, sector_cap_reached
)

# ==============================================================================
# REGIME CLASSIFIER (FIXING FLAW #2)
# Refactored to include volatility buffer buffers and momentum hysteresis to 
# prevent minor pullbacks from triggering false BEAR meltdowns.
# ==============================================================================
def classify_regime(sp500, sp500_sma200, sp500_sma50, prev_date, lookback_dates):
    """
    Returns ('BULL', 'BEAR', or 'CHOP') based on relaxed, institutionally-buffered
    trend, volatility-ratio, and directional momentum conditions.
    """
    try:
        price = sp500.loc[prev_date]
        sma200 = sp500_sma200.loc[prev_date]
        sma50 = sp500_sma50.loc[prev_date]
    except KeyError:
        return 'CHOP'

    above_200 = price > sma200
    above_50 = price > sma50

    # Calculate 20-day vs 200-day realized volatility distributions
    if len(lookback_dates) >= 21:
        recent_slice = sp500.loc[lookback_dates[-21:]]
        recent_returns = recent_slice.pct_change().dropna()
        recent_vol = recent_returns.std() * np.sqrt(252) if len(recent_returns) > 1 else 0.15
    else:
        recent_vol = 0.15

    if len(lookback_dates) >= 201:
        long_slice = sp500.loc[lookback_dates[-201:]]
        long_returns = long_slice.pct_change().dropna()
        long_vol = long_returns.std() * np.sqrt(252) if len(long_returns) > 1 else 0.15
    else:
        long_vol = 0.15

    # FIXED: Raised volatility ceiling trigger to 1.8x (old was 1.5x) to ignore 
    # standard bull market noise and prevent premature BEAR regime switches.
    vol_elevated = recent_vol > (long_vol * 1.8)

    # 20-day directional price momentum vector
    if len(lookback_dates) >= 21:
        momentum_pos = sp500.loc[lookback_dates[-1]] > sp500.loc[lookback_dates[-21]]
    else:
        momentum_pos = above_50

    # REFACTORED REGIME LOGIC GATEWAY:
    if above_200:
        if above_50 and momentum_pos and not vol_elevated:
            return 'BULL'
        else:
            return 'CHOP'  # Standard pullbacks transition to rangebound CHOP, not panic BEAR
    else:
        if vol_elevated or not momentum_pos:
            return 'BEAR'  # Hard BEAR only under broken macro trend + elevated distress
        else:
            return 'CHOP'


# ==============================================================================
# SLOT BUDGET ALLOCATOR (FIXING FLAW #3)
# Completely removes the max(1, int(v)) leakage. Allows unwanted strategies
# to drop to absolute ZERO slots to maximize specializations.
# ==============================================================================
def allocate_slots(regime, total_slots):
    """
    Returns {'Yang': n, 'Zhan': n, 'Yin': n} slot budgets.
    Enforces extreme specialization targets across target regimes.
    """
    if regime == 'BULL':
        # Yin is completely blacklisted (0.0) so it cannot dilute structural bull momentum
        weights = {'Yang': 0.70, 'Zhan': 0.30, 'Yin': 0.00}
    elif regime == 'BEAR':
        # Yang is completely blacklisted (0.0) to prevent catching falling knives in liquidations
        weights = {'Yang': 0.00, 'Zhan': 0.40, 'Yin': 0.60}
    else:  # CHOP
        weights = {'Yang': 0.15, 'Zhan': 0.55, 'Yin': 0.30}

    # FIXED: Floor changed to 0 instead of 1 to allow complete asset exclusion
    slots = {name: max(0, int(total_slots * w)) for name, w in weights.items()}

    # Distribute any remaining fractional rounding slots to the dominant strategy
    deficit = total_slots - sum(slots.values())
    if deficit > 0:
        top = max(weights, key=weights.get)
        slots[top] += deficit
    elif deficit < 0:
        # If rounding overshot, trim from the lowest active strategy (that has > 0 slots)
        active_weights = {k: v for k, v in weights.items() if slots[k] > 0}
        if active_weights:
            bottom = min(active_weights, key=active_weights.get)
            slots[bottom] = max(0, slots[bottom] + deficit)

    return slots


# ==============================================================================
# PORTFOLIO-LEVEL DRAWDOWN GUARD & VOL SCALER (FIXING FLAW #4 & #5)
# Replaced strict chokeholds with expanded institutional risk boundaries
# and continuous non-linear volatility target multipliers.
# ==============================================================================
def drawdown_guard(equity_curve, current_vol=None, target_vol=0.10):
    """
    Returns (entry_allowed: bool, size_mult: float).
    Allows the portfolio to breathe through routine equity curve fluctuations.
    """
    if len(equity_curve) < 2:
        return True, 1.0
        
    peak = max(equity_curve)
    current = equity_curve[-1]
    dd = (current - peak) / peak if peak > 0 else 0.0

    # FIXED FLAW #4: Expanded risk bands to prevent premature trade choking
    if dd < -0.25:
        entry_allowed, dd_mult = False, 0.0   # Hard circuit breaker at structural macro crisis (-25%)
    elif dd < -0.15:
        entry_allowed, dd_mult = True, 0.5    # Halve size only after significant trend drawdown (-15%)
    elif dd < -0.08:
        entry_allowed, dd_mult = True, 0.85   # Minor defensive adjustments at normal resistance pullbacks (-8%)
    else:
        entry_allowed, dd_mult = True, 1.0    # Full structural sizing capacity across standard market noise

    # FIXED FLAW #5: Dynamic Continuous Volatility Target Scaling
    # Expands maximum sizing clamp up to 2.5x to capture outperformance when vol is compressed
    if current_vol is not None and current_vol > 0:
        vol_mult = min(max(target_vol / current_vol, 0.25), 2.5)
    else:
        vol_mult = 1.0

    # Combine drawdown circuit mult and dynamic vol scale mult vectorially
    combined_size_mult = dd_mult * vol_mult

    return entry_allowed, combined_size_mult

# ==============================================================================
# VOL SCALER (DEPRECATED & MERGED WITH INTEGRATED PORTFOLIO DRAWDOWNS)
# ==============================================================================
def rolling_stats(series, window=30):
    s = pd.Series(series)
    if len(s) < window:
        return {'mean': 0.0, 'std': 0.0}
    roll = s.iloc[-window:]
    return {'mean': roll.mean(), 'std': roll.std()}

def compute_vol_scale(portfolio_daily_returns, target_vol=0.12, window=30):
    """
    DEPRECATED OVERLAPPING LOGIC: Dynamic non-linear portfolio volatility math 
    is now computed natively inside the unified drawdown_guard circuit layout matrix 
    to eliminate compounding parameter compression penalties.
    """
    return 1.0


# ==============================================================================
# CANDIDATE SIGNAL GENERATORS (POINT-IN-TIME DYNAMIC UPGRADES)
# Refactored to scan the true running indicator dictionary layout grid.
# ==============================================================================
def yang_candidates(ind, tickers, sectors, prev_date, open_positions, just_exited_today, ctx):
    candidates = []
    
    # FIXED: Iterate strictly through available dynamic keys to prevent data-gap crashes
    for t in list(ind.keys()):
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

        if pd.isna(y_close) or pd.isna(y_sma50) or pd.isna(y_atr) or y_atr <= 0:
            continue

        sector = sectors.get(t, "Unassigned") # Crash protection fallback
        if sector_cap_reached(open_positions, sector, ctx):
            continue

        if y_close > y_sma50 and pd.notna(y_vol) and pd.notna(y_avgvol30) and y_vol > y_avgvol30 and pd.notna(y_prior_high) and y_close > y_prior_high:
            strength = (y_close - y_prior_high) / y_atr
            candidates.append({'ticker': t, 'strategy': 'Yang', 'atr': y_atr,
                                'sector': sector, 'stop_mult': 2.0, 'strength': strength,
                                'mod_type': 'Yang Module A (High Vol Breakout)'})
        elif y_close > y_sma50 and pd.notna(y_vol) and pd.notna(y_avgvol30) and y_vol < y_avgvol30 and pd.notna(y_sma20) and y_close <= (y_sma20 + 2 * y_atr):
            strength = (y_sma50 - y_close) / y_atr
            candidates.append({'ticker': t, 'strategy': 'Yang', 'atr': y_atr,
                                'sector': sector, 'stop_mult': 2.0, 'strength': strength,
                                'mod_type': 'Yang Module B (Low Vol Pullback)'})
    return candidates

def yin_candidates(ind, tickers, sectors, prev_date, open_positions, just_exited_today, ctx):
    candidates = []
    
    # FIXED: Iterate strictly through available dynamic keys to prevent data-gap crashes
    for t in list(ind.keys()):
        if t in open_positions or t in just_exited_today:
            continue
        df = ind.get(t)
        if df is None or prev_date not in df.index:
            continue

        y_close = df.loc[prev_date, 'Close']
        y_rsi2 = df.loc[prev_date, 'RSI2']
        y_sma200 = df.loc[prev_date, 'SMA200']
        y_atr14 = df.loc[prev_date, 'ATR14']
        y_atr50 = df.loc[prev_date, 'ATR50'] if 'ATR50' in df.columns else None

        if pd.isna(y_close) or pd.isna(y_rsi2) or pd.isna(y_sma200) or pd.isna(y_atr14):
            continue

        sector = sectors.get(t, "Unassigned") # Crash protection fallback
        if sector_cap_reached(open_positions, sector, ctx):
            continue

        if y_atr50 is not None and pd.notna(y_atr50) and y_atr50 > 0 and y_atr14 > 2 * y_atr50:
            continue

        if y_close > y_sma200 and y_rsi2 < 5:
            strength = 15.0 - y_rsi2
            candidates.append({'ticker': t, 'strategy': 'Yin', 'atr': y_atr14,
                                'sector': sector, 'stop_mult': 1.0, 'strength': strength,
                                'mod_type': 'Yin Module B (Deep RSI < 5)'})
        elif y_close > y_sma200 and y_rsi2 < 10:
            strength = 10.0 - y_rsi2
            candidates.append({'ticker': t, 'strategy': 'Yin', 'atr': y_atr14,
                                'sector': sector, 'stop_mult': 1.0, 'strength': strength,
                                'mod_type': 'Yin Module A (Standard RSI < 10)'})
    return candidates

def zhan_candidates(ind, tickers, sectors, prev_date, open_positions, just_exited_today, ctx):
    candidates = []
    
    # FIXED: Iterate strictly through available dynamic keys to prevent data-gap crashes
    for t in list(ind.keys()):
        if t in open_positions or t in just_exited_today:
            continue
        df = ind.get(t)
        if df is None or prev_date not in df.index:
            continue

        try:
            curr_idx = df.index.get_loc(prev_date)
        except KeyError:
            continue
        if curr_idx < 4:
            continue

        y_close = df.loc[prev_date, 'Close']
        y_ema20 = df.loc[prev_date, 'EMA20']
        y_kc_upper = df.loc[prev_date, 'KC_UPPER']
        y_vol = df.loc[prev_date, 'Vol']
        y_avgvol30 = df.loc[prev_date, 'AvgVol30']
        y_atr = df.loc[prev_date, 'ATR14']

        if pd.isna(y_close) or pd.isna(y_atr) or y_atr <= 0:
            continue

        sector = sectors.get(t, "Unassigned") # Crash protection fallback
        if sector_cap_reached(open_positions, sector, ctx):
            continue

        c_close = df['Close'].values
        c_upper = df['KC_UPPER'].values
        c_lower = df['KC_LOWER'].values

        if any(np.isnan(c_close[curr_idx-k]) or np.isnan(c_upper[curr_idx-k]) or
               np.isnan(c_lower[curr_idx-k]) for k in range(1, 4)):
            continue

        cons = all(
            c_close[curr_idx-k] < c_upper[curr_idx-k] and c_close[curr_idx-k] > c_lower[curr_idx-k]
            for k in range(1, 4)
        )

        if y_close > y_kc_upper and pd.notna(y_vol) and pd.notna(y_avgvol30) and y_vol > y_avgvol30 and cons:
            strength = (y_close - y_kc_upper) / y_atr
            candidates.append({'ticker': t, 'strategy': 'Zhan', 'atr': y_atr,
                                'sector': sector, 'stop_mult': 1.5, 'strength': strength,
                                'mod_type': 'Zhan Module A (Keltner Breakout)'})
        elif y_close > y_ema20 and pd.notna(y_vol) and pd.notna(y_avgvol30) and y_vol < y_avgvol30 and cons:
            strength = (y_close - y_ema20) / y_atr
            candidates.append({'ticker': t, 'strategy': 'Zhan', 'atr': y_atr,
                                'sector': sector, 'stop_mult': 1.5, 'strength': strength,
                                'mod_type': 'Zhan Module B (EMA Breakout)'})
    return candidates

# ==============================================================================
# MAIN ENSEMBLE ENGINE (POINT-IN-TIME VERSION 3 REFACTOR)
# ==============================================================================
def run_ensemble_portfolio_core(ind, data, sp500, sp500_sma200, sp500_sma50, config_params, ctx, universe_history=None):
    """
    Executes the dynamic Point-in-Time adaptive Ensemble suite loop.
    Accepts the optional historical Wikipedia index change database arrays.
    """
    cash = ctx.START_CAPITAL
    equity = ctx.START_CAPITAL
    open_positions = {}
    trade_log = []
    equity_curve = []
    daily_returns = []
    portfolio_daily = []

    dates = sp500.index[200:]
    market_ok_series = (sp500 > sp500_sma200) & (sp500 > sp500_sma50)

    target_vol = config_params.get('target_vol', 0.12)

    for i in range(1, len(dates)):
        date = dates[i]
        prev_date = dates[i-1]
        lookback_dates = dates[:i]

        cash = apply_cash_drag(cash, ctx)
        market_ok = market_ok_series.get(prev_date, False)
        just_exited_today = set()

        # ------------------------------------------------------------------
        # REGIME CLASSIFICATION
        # ------------------------------------------------------------------
        regime = classify_regime(sp500, sp500_sma200, sp500_sma50, prev_date, lookback_dates)

        # ------------------------------------------------------------------
        # PORTFOLIO DRAWDOWN GUARD & CONTINUOUS VOL TARGET COUPLER
        # Calculates combined size scaling factor natively inside single layout pass
        # ------------------------------------------------------------------
        current_annual_vol = None
        if len(daily_returns) >= 30:
            roll_std = pd.Series(daily_returns).iloc[-30:].std()
            if pd.notna(roll_std) and roll_std > 0:
                current_annual_vol = roll_std * np.sqrt(252)

        entries_allowed, combined_size_mult = drawdown_guard(
            equity_curve, 
            current_vol=current_annual_vol, 
            target_vol=target_vol
        )

        # ------------------------------------------------------------------
        # PHASE A: EXITS — strategy-specific rules
        # ------------------------------------------------------------------
        to_close = []
        for t, pos in open_positions.items():
            df = ind.get(t)
            if df is None or prev_date not in df.index:
                continue

            prev_close = df.loc[prev_date, 'Close']
            y_atr14 = df.loc[prev_date, 'ATR14']
            strategy = pos['strategy']

            if strategy == 'Yin':
                prev_sma5 = df.loc[prev_date, 'SMA5']
                if pos['days_held'] >= 4 or prev_close > prev_sma5 or prev_close <= pos['stop']:
                    to_close.append(t)
            elif strategy == 'Yang':
                prev_sma50 = df.loc[prev_date, 'SMA50']
                if pd.notna(prev_close) and pd.notna(y_atr14) and y_atr14 > 0:
                    new_stop = prev_close - (pos['stop_mult'] * y_atr14)
                    if new_stop > pos['stop']:
                        pos['stop'] = new_stop
                if prev_close < prev_sma50 or prev_close <= pos['stop']:
                    to_close.append(t)
            elif strategy == 'Zhan':
                prev_kc_lower = df.loc[prev_date, 'KC_LOWER']
                if pd.notna(prev_close) and pd.notna(y_atr14) and y_atr14 > 0:
                    new_stop = prev_close - (pos['stop_mult'] * y_atr14)
                    if new_stop > pos['stop']:
                        pos['stop'] = new_stop
                if prev_close < prev_kc_lower or prev_close <= pos['stop']:
                    to_close.append(t)

        for t in to_close:
            pos = open_positions.pop(t)
            just_exited_today.add(t)
            if t not in data['Open'] or date not in data['Open'][t].index:
                continue

            open_price = data['Open'][t].loc[date]
            
            # FIXED: Safe forward-fill lookup logic for missing date indexes
            if t in ind and date in ind[t].index:
                close_price = ind[t].loc[date, 'Close']
            else:
                idx = ind[t].index if t in ind else []
                avail = idx[idx <= date] if len(idx) > 0 else []
                close_price = ind[t].loc[avail[-1], 'Close'] if len(avail) > 0 else open_price
                
            exit_price = gap_down_exit(open_price, pos['stop'], close_price)
            exit_price = apply_exit_friction(exit_price, ctx)
            exit_price = max(exit_price, 0.0001)

            cash += pos['shares'] * exit_price - ctx.COMMISSION
            pnl = (exit_price - pos['entry']) * pos['shares'] - ctx.COMMISSION
            trade_log.append({
                'ticker': t, 'entry_date': pos['entry_date'], 'exit_date': date,
                'entry_price': pos['entry'], 'exit_price': exit_price,
                'shares': pos['shares'], 'pnl': pnl, 'strategy': pos['strategy'],
                'regime': pos.get('entry_regime', 'UNKNOWN')
            })

        # ------------------------------------------------------------------
        # PHASE B: ENTRIES — regime-rationed slot budget
        # ------------------------------------------------------------------
        if market_ok and entries_allowed:
            total_available = ctx.MAX_TOTAL_POSITIONS - len(open_positions)
            slot_budget = allocate_slots(regime, ctx.MAX_TOTAL_POSITIONS)

            # How many slots each strategy *still* has left
            # (subtract currently open positions for that strategy)
            open_by_strategy = {}
            for pos in open_positions.values():
                s = pos['strategy']
                open_by_strategy[s] = open_by_strategy.get(s, 0) + 1

            remaining_budget = {
                name: max(0, slot_budget[name] - open_by_strategy.get(name, 0))
                for name in ['Yang', 'Zhan', 'Yin']
            }

            # FIXED: Directly leverage the integrated non-linear multiplier computed in Part 3
            final_size_mult = combined_size_mult

            if total_available > 0 and sum(remaining_budget.values()) > 0:
                # Generate candidates per strategy
                # FIXED: Swapped the undefined 'tickers' variable name out for list(ind.keys())
                yang_c = yang_candidates(ind, list(ind.keys()), sectors, prev_date, open_positions, just_exited_today, ctx)
                yin_c = yin_candidates(ind, list(ind.keys()), sectors, prev_date, open_positions, just_exited_today, ctx)
                zhan_c = zhan_candidates(ind, list(ind.keys()), sectors, prev_date, open_positions, just_exited_today, ctx)


                # Sort each pool by their own strength score
                yang_c.sort(key=lambda x: x['strength'], reverse=True)
                yin_c.sort(key=lambda x: x['strength'], reverse=True)
                zhan_c.sort(key=lambda x: x['strength'], reverse=True)

                # Allocate from each pool up to their remaining budget,
                # round-robin style so no single strategy monopolizes a day
                strategy_queues = {
                    'Yang': yang_c[:remaining_budget['Yang']],
                    'Yin': yin_c[:remaining_budget['Yin']],
                    'Zhan': zhan_c[:remaining_budget['Zhan']]
                }

                # Interleave: take top-1 from each strategy in priority order,
                # then loop until budgets exhausted or slots filled
                priority_order = (
                    ['Yang', 'Zhan', 'Yin'] if regime == 'BULL' else
                    ['Yin', 'Zhan', 'Yang'] if regime == 'BEAR' else
                    ['Zhan', 'Yin', 'Yang']
                )

                queue_indices = {name: 0 for name in priority_order}
                slots_used = 0

                # FIXED: Establish static baseline pool mapping for unified entry waterfall sizing
                sizing_equity = equity

                while slots_used < total_available:
                    filled_this_pass = 0
                    for name in priority_order:
                        if slots_used >= total_available:
                            break
                        idx = queue_indices[name]
                        pool = strategy_queues[name]
                        if idx >= len(pool):
                            continue

                        c = pool[idx]
                        queue_indices[name] += 1
                        t = c['ticker']

                        if t in open_positions or t in just_exited_today:
                            continue
                        
                        sector = sectors.get(t, 'Unassigned') # Robust dictionary protection hook
                        if sector_cap_reached(open_positions, sector, ctx):
                            continue
                        if t not in data['Open'] or date not in data['Open'][t].index:
                            continue

                        raw_open = data['Open'][t].loc[date]
                        if pd.isna(raw_open) or raw_open <= 0:
                            continue

                        stop = raw_open - (c['stop_mult'] * c['atr'])
                        
                        # FIXED: Size position leveraging unified portfolio remaining allocation cash capacity
                        base_shares = position_size(raw_open, stop, sizing_equity, ctx)
                        shares = max(1, int(base_shares * final_size_mult)) if base_shares > 0 else 0
                        if shares <= 0:
                            continue

                        entry_price = apply_entry_friction(raw_open, ctx)
                        total_cost = (shares * entry_price) + ctx.COMMISSION

                        if cash >= total_cost:
                            cash -= total_cost
                            open_positions[t] = {
                                'entry': entry_price, 'stop': stop, 'shares': shares,
                                'sector': sector, 'entry_date': date,
                                'strategy': c['strategy'], 'stop_mult': c['stop_mult'],
                                'days_held': 0, 'entry_regime': regime
                            }
                            slots_used += 1
                            filled_this_pass += 1
                            # FIXED: Account for dynamic capital drainage inside waterfall allocation loop
                            sizing_equity -= total_cost

                    if filled_this_pass == 0:
                        break  # Nothing left to allocate across all queues

        # ------------------------------------------------------------------
        # BOOKKEEPING
        # ------------------------------------------------------------------
        for t in open_positions:
            open_positions[t]['days_held'] += 1

        open_val = 0.0
        for t, pos in open_positions.items():
            df = ind.get(t)
            if df is not None:
                available = df.index[df.index <= date]
                price = df.loc[available[-1], 'Close'] if not available.empty else pos['entry']
            else:
                price = pos['entry']
            open_val += pos['shares'] * price

        prev_equity = equity
        equity = cash + open_val

        # FIXED: Log returns for every single day to guarantee strict matrix symmetry
        ret = (equity - prev_equity) / prev_equity
        daily_returns.append(ret)
        portfolio_daily.append(ret)
        equity_curve.append(equity)

    return {
        'name': 'Ensemble', 'equity_curve': equity_curve, 'trade_log': trade_log,
        'daily_returns': daily_returns, 'dates': list(dates[1:])
    }


# ==============================================================================
# SUITE WRAPPER (PIT REFACTOR SANITY LOCK)
# ==============================================================================
def run_ensemble_suite(ind, data, sp500, sp500_sma200, sp500_sma50, config_params, universe_history=None):
    """
    Main interface gateway. Explicitly assigns keyword arguments to prevent
    positional matrix collisions down the compilation pipeline.
    """
    ctx = SystemContext(
        max_total_positions=config_params.get('max_total_positions', 8),
        max_alloc_fraction=config_params.get('max_alloc_fraction', 1/6),
        max_sector_positions=config_params.get('max_sector_positions', 2),
        target_vol=config_params.get('target_vol', 0.12)
    )
    
    # Direct explicit mapping to your logic core function block name
    return run_ensemble_portfolio_core(
        ind, data, sp500, sp500_sma200, sp500_sma50,
        config_params=config_params, 
        ctx=ctx, 
        universe_history=universe_history
    )
