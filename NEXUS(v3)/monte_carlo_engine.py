import numpy as np
import pandas as pd
from config_and_data import SystemContext

# ==============================================================================
# MONTE CARLO - VECTORIZED LOG-NORMAL STATIONARY BLOCK BOOTSTRAP
# ==============================================================================
def monte_carlo_bootstrap(daily_returns, ctx: SystemContext, n_sims=5000, block_size=5, seed=42):
    rng = np.random.default_rng(seed)
    daily = np.asarray(daily_returns, dtype=float)
    n = len(daily)
    if n == 0:
        return None
    block_size = min(block_size, n)

    num_blocks_needed = int(np.ceil(n / block_size))
    max_start_idx = n - block_size + 1 if n > block_size else 1

    start_indices = rng.integers(0, max_start_idx, size=(n_sims, num_blocks_needed))
    sim_paths = np.zeros((n_sims, num_blocks_needed * block_size))
    
    for block_offset in range(block_size):
        sim_paths[:, block_offset::block_size] = daily[start_indices + block_offset]
    sim_paths = sim_paths[:, :n]

    # Dynamically extract start capital from the live runtime context container
    log_sim_paths = np.log1p(sim_paths)
    eq_paths = ctx.START_CAPITAL * np.exp(np.cumsum(log_sim_paths, axis=1))
    
    running_peaks = np.maximum.accumulate(eq_paths, axis=1)
    drawdowns = (eq_paths - running_peaks) / running_peaks
    max_dds = np.min(drawdowns, axis=1) * 100
    total_returns = (eq_paths[:, -1] - ctx.START_CAPITAL) / ctx.START_CAPITAL * 100

    # Dynamically extract the daily risk-free parameter index safely from context
    means = np.mean(sim_paths - ctx.DAILY_RF, axis=1)
    stds = np.std(sim_paths, axis=1)
    sharpes = np.where(stds > 0, (means / stds) * np.sqrt(252), 0.0)

    return {'total_return': total_returns, 'max_dd': max_dds, 'sharpe': sharpes}

# ==============================================================================
# MULTI-STRATEGY MONTE CARLO TORTURE TESTING UTILITY
# ==============================================================================
def run_monte_carlo_suite(results, ctx: SystemContext, n_sims=5000, block_size=5, seed=42):
    summaries = []
    mc_by_name = {}

    for r in results:
        actual_total_return = (pd.Series(r['equity_curve']).iloc[-1] - ctx.START_CAPITAL) / ctx.START_CAPITAL * 100
        mc = monte_carlo_bootstrap(r['daily_returns'], ctx, n_sims=n_sims, block_size=block_size, seed=seed)
        if mc is None:
            continue
        mc_by_name[r['name']] = mc
        summaries.append({
            'name': r['name'],
            'return_p5': np.percentile(mc['total_return'], 5),
            'return_p25': np.percentile(mc['total_return'], 25),
            'return_median': np.percentile(mc['total_return'], 50),
            'return_p75': np.percentile(mc['total_return'], 75),
            'return_p95': np.percentile(mc['total_return'], 95),
            'dd_p5': np.percentile(mc['max_dd'], 5),
            'sharpe_median': np.percentile(mc['sharpe'], 50),
            'pct_sims_positive': (mc['total_return'] > 0).mean() * 100,
            'actual_return': actual_total_return,
        })

    table = pd.DataFrame(summaries)
    print("\n==============================================================")
    print(f"MONTE CARLO (block bootstrap, {n_sims} sims, block size {block_size} days)")
    print("==============================================================")
    
    print_cols = ['name', 'return_p5', 'return_p25', 'return_median', 'return_p75',
                  'return_p95', 'dd_p5', 'sharpe_median', 'pct_sims_positive', 'actual_return']
    
    rename = {
        'name': 'Model', 'return_p5': 'Return P5%', 'return_p25': 'Return P25%',
        'return_median': 'Return Median%', 'return_p75': 'Return P75%',
        'return_p95': 'Return P95%', 'dd_p5': 'Tail Risk DD% (P5)',
        'sharpe_median': 'Median Sharpe', 'pct_sims_positive': '% Sims Profitable',
        'actual_return': 'Actual Return%'
    }
    
    print(table[print_cols].rename(columns=rename).round(2).to_string(index=False))

    if 'SPX Buy & Hold' in mc_by_name:
        bench_tr = mc_by_name['SPX Buy & Hold']['total_return']
        print("\n==============================================================")
        print("P(STRATEGY PATH BEATS SPX PATH) -- independently resampled Monte Carlo runs")
        print("==============================================================")
        for name, mc in mc_by_name.items():
            if name == 'SPX Buy & Hold':
                continue
            n = min(len(mc['total_return']), len(bench_tr))
            win_rate = (mc['total_return'][:n] > bench_tr[:n]).mean() * 100
            print(f"{name:>10}: beats SPX in {win_rate:.1f}% of paired simulations")

    return table
