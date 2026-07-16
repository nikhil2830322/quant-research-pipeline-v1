import numpy as np
import pandas as pd
from config_and_data import SystemContext, tickers, sectors
from risk_and_friction import (
    apply_cash_drag, apply_exit_friction, apply_entry_friction,
    gap_down_exit, position_size, sector_cap_reached
)

# ==============================================================================
# REGIME CLASSIFIER
# Three-state market regime: BULL / BEAR / CHOP
# This drives which strategies get slots and how many. The key insight from
# backtesting: Yang outperforms in bull, underperforms in chop/bear. Yin
# outperforms in bear/volatile, is mostly idle in bull. Zhan is the most
# regime-agnostic. So instead of running all three equally, we ration slots
# by regime to lean into each strategy's structural advantage.
# ==============================================================================
def classify_regime(sp500, sp500_sma200, sp500_sma50, prev_date, lookback_dates):
    """
    Returns ('BULL', 'BEAR', or 'CHOP') based on three signals:
      - Price vs SMA200 and SMA50 (trend direction)
      - 20-day realized volatility vs long-run vol (vol regime)
      - Recent momentum: 20-day return direction
    """
    try:
        price = sp500.loc[prev_date]
        sma200 = sp500_sma200.loc[prev_date]
        sma50 = sp500_sma50.loc[prev_date]
    except KeyError:
        return 'CHOP'

    above_200 = price > sma200
    above_50 = price > sma50

    # 20-day realized vol vs 200-day realized vol
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

    vol_elevated = recent_vol > (long_vol * 1.5)

    # 20-day momentum
    if len(lookback_dates) >= 21:
        momentum_pos = sp500.loc[lookback_dates[-1]] > sp500.loc[lookback_dates[-21]]
    else:
        momentum_pos = above_50

    if above_200 and above_50 and not vol_elevated and momentum_pos:
        return 'BULL'
    elif (not above_200) or vol_elevated:
        return 'BEAR'
    else:
        return 'CHOP'


# ==============================================================================
# SLOT BUDGET ALLOCATOR
# Given the regime, returns how many of the total available slots each
# strategy is allowed to use. This is the core rationing mechanic.
# Strategies don't compete equally — the regime tilts the playing field.
# ==============================================================================
def allocate_slots(regime, total_slots):
    """
    Returns {'Yang': n, 'Zhan': n, 'Yin': n} slot budgets.
    Total sums to total_slots (with rounding handled).
    
    BULL:  Yang gets the most (trend-following in a bull = structural edge),
           Zhan gets moderate, Yin gets minimal (MR into rising market = bad)
    BEAR:  Yin gets priority (mean reversion in oversold = structural edge),
           Zhan moderate (vol expansion still works), Yang gets minimal
           (trend-following into a crash = exactly wrong)
    CHOP:  Zhan gets most (volatility expansion = designed for this),
           Yin moderate (MR works in rangebound markets),
           Yang minimal (no clear trend = trend-following struggles)
    """
    if regime == 'BULL':
        weights = {'Yang': 0.55, 'Zhan': 0.30, 'Yin': 0.15}
    elif regime == 'BEAR':
        weights = {'Yang': 0.10, 'Zhan': 0.35, 'Yin': 0.55}
    else:  # CHOP
        weights = {'Yang': 0.20, 'Zhan': 0.50, 'Yin': 0.30}

    raw = {name: total_slots * w for name, w in weights.items()}
    # Floor each to at least 1 (no strategy gets completely zeroed out —
    # edge cases exist within regimes, so a minimal presence is kept)
    slots = {name: max(1, int(v)) for name, v in raw.items()}

    # Distribute any rounding remainder to the highest-weighted strategy
    deficit = total_slots - sum(slots.values())
    if deficit > 0:
        top = max(weights, key=weights.get)
        slots[top] += deficit
    elif deficit < 0:
        # If rounding overshot, trim from the lowest-weighted strategy
        bottom = min(weights, key=weights.get)
        slots[bottom] = max(1, slots[bottom] + deficit)

    return slots


# ==============================================================================
# PORTFOLIO-LEVEL DRAWDOWN GUARD
# If the overall portfolio is down more than the threshold from its peak,
# tighten the entry gate (fewer new positions) and shrink sizing.
# This acts as a circuit breaker across all three strategies simultaneously --
# something none of the individual models have, since each only manages its
# own stops without awareness of portfolio-level pain.
# ==============================================================================
def drawdown_guard(equity_curve):
    """
    Returns (entry_allowed: bool, size_mult: float).
    entry_allowed: False if portfolio DD > 15% -- pause all new entries
    size_mult: scales position size down as DD deepens
    """
    if len(equity_curve) < 2:
        return True, 1.0
    peak = max(equity_curve)
    current = equity_curve[-1]
    dd = (current - peak) / peak if peak > 0 else 0.0

    if dd < -0.15:
        return False, 0.0   # pause entries entirely
    elif dd < -0.08:
        return True, 0.5    # half-size entries
    elif dd < -0.04:
        return True, 0.75   # three-quarter-size entries
    else:
        return True, 1.0


# ==============================================================================
# VOL SCALER (clamped)
# ==============================================================================
def rolling_stats(series, window=30):
    s = pd.Series(series)
    if len(s) < window:
        return {'mean': 0.0, 'std': 0.0}
    roll = s.iloc[-window:]
    return {'mean': roll.mean(), 'std': roll.std()}

def compute_vol_scale(portfolio_daily_returns, target_vol=0.12, window=30):
    rs = rolling_stats(portfolio_daily_returns, window=window)
    if rs['std'] <= 0:
        return 1.0
    current_annual_vol = rs['std'] * np.sqrt(252)
    if current_annual_vol <= 0:
        return 1.0
    # Clamped to [0.5, 1.5]: vol targeting adjusts sizing but never
    # halves it more than 2x or doubles it -- keeps behavior predictable
    return min(max(target_vol / current_annual_vol, 0.5), 1.5)


# ==============================================================================
# CANDIDATE SIGNAL GENERATORS (per strategy)
# Each returns a list of candidate dicts with 'strength', 'ticker',
# 'strategy', 'atr', 'sector', 'stop_mult'. They are scored on their own
# scale, then the regime-weighted strength is used for cross-strategy ranking
# within each strategy's slot budget.
# ==============================================================================
def yang_candidates(ind, tickers, sectors, prev_date, open_positions, just_exited_today, ctx):
    candidates = []
    for t in tickers:
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

        sector = sectors.get(t, "Unknown")
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
    for t in tickers:
        if t in open_positions or t in just_exited_today:
            continue
        df = ind.get(t)
        if df is None or prev_date not in df.index:
            continue

        y_close = df.loc[prev_date, 'Close']
        y_rsi2 = df.loc[prev_date, 'RSI2']
        y_sma200 = df.loc[prev_date, 'SMA200']
        y_atr14 = df.loc[prev_date, 'ATR14']
        y_atr50 = df.loc[prev_date, 'ATR50']

        if pd.isna(y_close) or pd.isna(y_rsi2) or pd.isna(y_sma200) or pd.isna(y_atr14):
            continue

        sector = sectors.get(t, "Unknown")
        if sector_cap_reached(open_positions, sector, ctx):
            continue

        if pd.notna(y_atr50) and y_atr50 > 0 and y_atr14 > 2 * y_atr50:
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
    for t in tickers:
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

        sector = sectors.get(t, "Unknown")
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
# MAIN ENSEMBLE ENGINE
# ==============================================================================
def run_ensemble_portfolio_advanced(ind, data, sp500, sp500_sma200, sp500_sma50,
                                     config_params, ctx: SystemContext):
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
        # PORTFOLIO DRAWDOWN GUARD
        # ------------------------------------------------------------------
        entries_allowed, dd_size_mult = drawdown_guard(equity_curve)

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
            close_price = ind[t].loc[date, 'Close'] if date in ind[t].index else open_price
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

            vol_scale = compute_vol_scale(portfolio_daily, target_vol=target_vol)
            final_size_mult = dd_size_mult * vol_scale

            if total_available > 0 and sum(remaining_budget.values()) > 0:
                # Generate candidates per strategy
                yang_c = yang_candidates(ind, tickers, sectors, prev_date, open_positions, just_exited_today, ctx)
                yin_c = yin_candidates(ind, tickers, sectors, prev_date, open_positions, just_exited_today, ctx)
                zhan_c = zhan_candidates(ind, tickers, sectors, prev_date, open_positions, just_exited_today, ctx)

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
                        if sector_cap_reached(open_positions, sectors.get(t, 'Unknown'), ctx):
                            continue
                        if t not in data['Open'] or date not in data['Open'][t].index:
                            continue

                        raw_open = data['Open'][t].loc[date]
                        if pd.isna(raw_open) or raw_open <= 0:
                            continue

                        stop = raw_open - (c['stop_mult'] * c['atr'])
                        base_shares = position_size(raw_open, stop, equity, ctx)
                        shares = max(1, int(base_shares * final_size_mult)) if base_shares > 0 else 0
                        if shares <= 0:
                            continue

                        entry_price = apply_entry_friction(raw_open, ctx)
                        total_cost = (shares * entry_price) + ctx.COMMISSION

                        if cash >= total_cost:
                            cash -= total_cost
                            open_positions[t] = {
                                'entry': entry_price, 'stop': stop, 'shares': shares,
                                'sector': sectors.get(t, 'Unknown'), 'entry_date': date,
                                'strategy': c['strategy'], 'stop_mult': c['stop_mult'],
                                'days_held': 0, 'entry_regime': regime
                            }
                            slots_used += 1
                            filled_this_pass += 1

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

        if i > 1:
            ret = (equity - prev_equity) / prev_equity
            daily_returns.append(ret)
            portfolio_daily.append(ret)

        equity_curve.append(equity)

    return {
        'name': 'Ensemble', 'equity_curve': equity_curve, 'trade_log': trade_log,
        'daily_returns': daily_returns, 'dates': list(dates[1:])
    }


# ==============================================================================
# SUITE WRAPPER
# ==============================================================================
def run_ensemble_suite(ind, data, sp500, sp500_sma200, sp500_sma50, config_params):
    ctx = SystemContext(
        max_total_positions=config_params.get('max_total_positions', 8),
        max_alloc_fraction=config_params.get('max_alloc_fraction', 1/6),
        max_sector_positions=config_params.get('max_sector_positions', 2),
        target_vol=config_params.get('target_vol', 0.12)
    )
    return run_ensemble_portfolio_advanced(ind, data, sp500, sp500_sma200, sp500_sma50,
                                           config_params, ctx)