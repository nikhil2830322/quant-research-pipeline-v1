# ==============================================================================
# ENSEMBLE - dynamic Yin/Yang/Zhan blend with regime + performance weighting P. S THIS IS FLAWED STILL HAVE TO FIX
# ==============================================================================
def rolling_stats(series, window=30):
    s = pd.Series(series)
    if len(s) < window:
        return {'mean': 0.0, 'std': 0.0}
    roll = s.iloc[-window:]
    return {'mean': roll.mean(), 'std': roll.std()}

def compute_strategy_metrics(results_dict):
    metrics = {}
    for name, res in results_dict.items():
        daily = res.get('daily_returns', [])
        eq = res.get('equity_curve', [])
        if len(daily) == 0 or len(eq) == 0:
            metrics[name] = {'sharpe_30d': 0.0, 'dd': 0.0}
            continue
        rs = rolling_stats(daily, window=30)
        sharpe_30d = ((rs['mean'] - DAILY_RF) / rs['std']) * np.sqrt(252) if rs['std'] > 0 else 0.0
        eq_series = pd.Series(eq)
        peak = eq_series.cummax()
        dd_series = (eq_series - peak) / peak
        dd = dd_series.iloc[-1] if len(dd_series) > 0 else 0.0
        metrics[name] = {'sharpe_30d': sharpe_30d, 'dd': dd}
    return metrics

def build_strategy_weights(metrics):
    raw = {}
    for name, m in metrics.items():
        sharpe = max(m['sharpe_30d'], 0.0)
        dd = m['dd']
        if dd < -0.2:
            dd_mult = 0.1
        elif dd < -0.1:
            dd_mult = 0.5
        else:
            dd_mult = 1.0
        raw[name] = sharpe * dd_mult
    total = sum(raw.values())
    if total <= 0:
        return {name: 1/3 for name in raw.keys()}
    return {name: v / total for name, v in raw.items()}

def compute_vol_scale(portfolio_daily_returns, target_vol=0.10, window=30):
    rs = rolling_stats(portfolio_daily_returns, window=window)
    if rs['std'] <= 0:
        return 1.0
    current_annual_vol = rs['std'] * np.sqrt(252)
    if current_annual_vol <= 0:
        return 1.0
    return target_vol / current_annual_vol

def detect_vol_regime(sp500, dates, window=20):
    if len(dates) < window + 1:
        return 'medium'
    closes = sp500.loc[dates[-window:]]
    returns = closes.pct_change().dropna()
    if len(returns) == 0:
        return 'medium'
    vol = returns.std() * np.sqrt(252)
    if vol < 0.12:
        return 'low'
    elif vol < 0.20:
        return 'medium'
    else:
        return 'high'

def regime_strategy_weights(regime):
    if regime == 'low':
        return {'Yang': 1.4, 'Yin': 1.0, 'Zhan': 0.8}
    elif regime == 'medium':
        return {'Yang': 1.0, 'Yin': 1.0, 'Zhan': 1.0}
    else:
        return {'Yang': 0.7, 'Yin': 1.2, 'Zhan': 1.3}

def run_ensemble_portfolio_advanced(ind, data, sp500, sp500_sma200, sp500_sma50, config_params):
    """Multi-strategy ensemble with volatility targeting, regime-aware
    weighting, and performance-based (Sharpe + drawdown) reweighting.

    FIXED: strategy_results used to be initialized empty and never updated,
    which silently forced compute_strategy_metrics -> build_strategy_weights
    to fall back to flat 1/3 weights every single day, regardless of which
    strategy was actually performing better. Now each strategy's realized
    trade PnL is tracked as it happens and converted into a running proxy
    equity curve, so the Sharpe/drawdown-based weighting actually has real
    data to react to. This proxy only moves on days a trade in that
    strategy closes (not full mark-to-market every day) -- a reasonable
    approximation for weighting purposes, not a full attribution system.
    """
    cash = START_CAPITAL
    equity = START_CAPITAL
    open_positions = {}
    trade_log = []
    equity_curve = []
    daily_returns = []

    dates = sp500.index[200:]
    portfolio_daily = []

    strategy_names = ['Yin', 'Yang', 'Zhan']
    strategy_cum_pnl = {name: 0.0 for name in strategy_names}
    strategy_results = {
        name: {'daily_returns': [], 'equity_curve': [START_CAPITAL / 3]}
        for name in strategy_names
    }

    for i in range(1, len(dates)):
        date = dates[i]
        prev_date = dates[i-1]
        cash = apply_cash_drag(cash)

        market_ok = (sp500.loc[prev_date] > sp500_sma200.loc[prev_date]) and \
                    (sp500.loc[prev_date] > sp500_sma50.loc[prev_date])

        just_exited_today = set()

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
                calc_stop = prev_close - (pos['stop_mult'] * y_atr14)
                if calc_stop > pos['stop']:
                    pos['stop'] = calc_stop
                if prev_close < prev_sma50 or prev_close <= pos['stop']:
                    to_close.append(t)
            elif strategy == 'Zhan':
                prev_kc_lower = df.loc[prev_date, 'KC_LOWER']
                calc_stop = prev_close - (pos['stop_mult'] * y_atr14)
                if calc_stop > pos['stop']:
                    pos['stop'] = calc_stop
                if prev_close < prev_kc_lower or prev_close <= pos['stop']:
                    to_close.append(t)

        for t in to_close:
            pos = open_positions.pop(t)
            just_exited_today.add(t)
            if t not in data['Open'] or date not in data['Open'][t].index:
                continue
            open_price = data['Open'][t].loc[date]
            close_price = ind[t].loc[date, 'Close']
            exit_price = gap_down_exit(open_price, pos['stop'], close_price)
            exit_price = apply_exit_friction(exit_price)

            cash += pos['shares'] * exit_price
            cash -= COMMISSION

            pnl = (exit_price - pos['entry']) * pos['shares'] - COMMISSION
            trade_log.append({
                'ticker': t, 'entry_date': pos['entry_date'], 'exit_date': date,
                'entry_price': pos['entry'], 'exit_price': exit_price,
                'shares': pos['shares'], 'pnl': pnl, 'strategy': pos['strategy']
            })
            strategy_cum_pnl[pos['strategy']] += pnl

        for name in strategy_names:
            proxy_equity = (START_CAPITAL / 3) + strategy_cum_pnl[name]
            prev_proxy = strategy_results[name]['equity_curve'][-1]
            strategy_results[name]['equity_curve'].append(proxy_equity)
            if prev_proxy != 0:
                strategy_results[name]['daily_returns'].append((proxy_equity - prev_proxy) / prev_proxy)

        metrics = compute_strategy_metrics(strategy_results)
        base_weights = build_strategy_weights(metrics)
        regime = detect_vol_regime(sp500, dates[:i+1])
        regime_mult = regime_strategy_weights(regime)
        strategy_weight = {
            name: base_weights.get(name, 0.0) * regime_mult.get(name, 1.0)
            for name in strategy_names
        }
        vol_scale = compute_vol_scale(portfolio_daily, target_vol=config_params.get('target_vol', 0.10))

        if market_ok:
            available_slots = config_params['max_total_positions'] - len(open_positions)
            candidates = []

            for t in tickers:
                if t in open_positions or t in just_exited_today:
                    continue
                df = ind.get(t)
                if df is None or prev_date not in df.index:
                    continue

                sector = sectors.get(t, "Unknown")
                if sector_cap_reached(open_positions, sector):
                    continue

                y_close = df.loc[prev_date, 'Close']
                y_vol = df.loc[prev_date, 'Vol']
                y_avgvol30 = df.loc[prev_date, 'AvgVol30']
                y_atr14 = df.loc[prev_date, 'ATR14']

                y_rsi2 = df.loc[prev_date, 'RSI2']
                y_sma200 = df.loc[prev_date, 'SMA200']
                y_atr50 = df.loc[prev_date, 'ATR50']
                yin_vol_ok = not (pd.notna(y_atr50) and y_atr50 > 0 and y_atr14 > 2 * y_atr50)

                if yin_vol_ok and y_close > y_sma200 and y_rsi2 < 10:
                    strength = (15.0 - y_rsi2) if y_rsi2 < 5 else (10.0 - y_rsi2)
                    strength *= strategy_weight['Yin']
                    candidates.append({'ticker': t, 'strategy': 'Yin', 'atr': y_atr14,
                                        'sector': sector, 'stop_mult': 1.0, 'strength': strength})
                    continue

                y_sma50 = df.loc[prev_date, 'SMA50']
                y_prior_high = df.loc[prev_date, 'Prior20High']
                y_sma20 = df.loc[prev_date, 'SMA20']

                if y_close > y_sma50 and y_vol > y_avgvol30 and y_close > y_prior_high:
                    base_strength = (y_close - y_prior_high) / y_atr14 if y_atr14 > 0 else 0
                    strength = base_strength * strategy_weight['Yang']
                    candidates.append({'ticker': t, 'strategy': 'Yang', 'atr': y_atr14,
                                        'sector': sector, 'stop_mult': 2.0, 'strength': strength})
                    continue
                elif y_close > y_sma50 and y_vol < y_avgvol30 and y_close <= (y_sma20 + 2 * y_atr14):
                    base_strength = (y_sma50 - y_close) / y_atr14 if y_atr14 > 0 else 0
                    strength = base_strength * strategy_weight['Yang']
                    candidates.append({'ticker': t, 'strategy': 'Yang', 'atr': y_atr14,
                                        'sector': sector, 'stop_mult': 2.0, 'strength': strength})
                    continue

                try:
                    curr_idx = df.index.get_loc(prev_date)
                except KeyError:
                    continue
                if curr_idx < 4:
                    continue

                y_ema20 = df.loc[prev_date, 'EMA20']
                y_kc_upper = df.loc[prev_date, 'KC_UPPER']
                cons = (
                    df.iloc[curr_idx-1]['Close'] < df.iloc[curr_idx-1]['KC_UPPER'] and
                    df.iloc[curr_idx-1]['Close'] > df.iloc[curr_idx-1]['KC_LOWER'] and
                    df.iloc[curr_idx-2]['Close'] < df.iloc[curr_idx-2]['KC_UPPER'] and
                    df.iloc[curr_idx-2]['Close'] > df.iloc[curr_idx-2]['KC_LOWER'] and
                    df.iloc[curr_idx-3]['Close'] < df.iloc[curr_idx-3]['KC_UPPER'] and
                    df.iloc[curr_idx-3]['Close'] > df.iloc[curr_idx-3]['KC_LOWER']
                )

                if y_close > y_kc_upper and y_vol > y_avgvol30 and cons:
                    base_strength = (y_close - y_kc_upper) / y_atr14 if y_atr14 > 0 else 0
                    strength = base_strength * strategy_weight['Zhan']
                    candidates.append({'ticker': t, 'strategy': 'Zhan', 'atr': y_atr14,
                                        'sector': sector, 'stop_mult': 1.5, 'strength': strength})
                elif y_close > y_ema20 and y_vol < y_avgvol30 and cons:
                    base_strength = (y_close - y_ema20) / y_atr14 if y_atr14 > 0 else 0
                    strength = base_strength * strategy_weight['Zhan']
                    candidates.append({'ticker': t, 'strategy': 'Zhan', 'atr': y_atr14,
                                        'sector': sector, 'stop_mult': 1.5, 'strength': strength})

            candidates.sort(key=lambda x: x['strength'], reverse=True)
            transient_equity = equity

            for c in candidates:
                if available_slots <= 0:
                    break
                t = c['ticker']
                sector = c['sector']
                if sector_cap_reached(open_positions, sector):
                    continue
                if t not in data['Open'] or date not in data['Open'][t].index:
                    continue

                raw_open = data['Open'][t].loc[date]
                stop = raw_open - (c['stop_mult'] * c['atr'])
                base_shares = position_size(raw_open, stop, transient_equity)
                shares = int(base_shares * vol_scale)
                entry_price = apply_entry_friction(raw_open)
                total_cost = (shares * entry_price) + COMMISSION

                if shares > 0 and cash >= total_cost:
                    cash -= total_cost
                    open_positions[t] = {
                        'entry': entry_price, 'stop': stop, 'shares': shares,
                        'sector': sector, 'entry_date': date, 'strategy': c['strategy'],
                        'stop_mult': c['stop_mult'], 'days_held': 0
                    }
                    available_slots -= 1
                    transient_equity -= total_cost

        for t in open_positions:
            open_positions[t]['days_held'] += 1

        open_val = 0.0
        for t, pos in open_positions.items():
            df = ind.get(t)
            if df is not None and date in df.index:
                open_val += pos['shares'] * df.loc[date, 'Close']
            else:
                open_val += pos['shares'] * pos['entry']

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

def run_ensemble_suite(ind, data, sp500, sp500_sma200, sp500_sma50, config_params):
    """Explicit wrapper: temporarily sets MAX_TOTAL_POSITIONS / MAX_ALLOC_FRACTION
    / MAX_SECTOR_POSITIONS from config_params (instead of silently relying on
    whatever the module-level globals happened to be left at from a prior
    run), then restores them. Mirrors run_full_suite's pattern."""
    global MAX_TOTAL_POSITIONS, MAX_ALLOC_FRACTION, MAX_SECTOR_POSITIONS
    orig = (MAX_TOTAL_POSITIONS, MAX_ALLOC_FRACTION, MAX_SECTOR_POSITIONS)
    MAX_TOTAL_POSITIONS = config_params.get('max_total_positions', 8)
    MAX_ALLOC_FRACTION = config_params.get('max_alloc_fraction', 1/6)
    MAX_SECTOR_POSITIONS = config_params.get('max_sector_positions', 2)

    result = run_ensemble_portfolio_advanced(ind, data, sp500, sp500_sma200, sp500_sma50, config_params)

    MAX_TOTAL_POSITIONS, MAX_ALLOC_FRACTION, MAX_SECTOR_POSITIONS = orig
    return result
