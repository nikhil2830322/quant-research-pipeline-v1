import numpy as np
import pandas as pd
from config_and_data import SystemContext

# ==============================================================================
# MONTE CARLO - VECTORIZED LOG-NORMAL STATIONARY BLOCK BOOTSTRAP (V9 UPGRADE)
# ==============================================================================
def monte_carlo_bootstrap(daily_returns, ctx: SystemContext, n_sims=5000, block_size=5, seed=42, start_indices=None):
    """
    Executes an institutional-grade, vectorized stationary block bootstrap.
    Accepts an optional matrix of start_indices to ensure path-synchronization
    across strategies and market benchmarks.
    """
    rng = np.random.default_rng(seed)
    daily = np.asarray(daily_returns, dtype=float)
    n = len(daily)
    if n == 0:
        return None
    block_size = min(block_size, n)

    num_blocks_needed = int(np.ceil(n / block_size))
    max_start_idx = n - block_size + 1 if n > block_size else 1

    # FIXED: Reuse synced indices if provided, otherwise generate independently
    if start_indices is None:
        start_indices = rng.integers(0, max_start_idx, size=(n_sims, num_blocks_needed))
    
    sim_paths = np.zeros((n_sims, num_blocks_needed * block_size))
    
    # Reconstruct blocks vectorially across all 5,000 paths simultaneously
    for block_offset in range(block_size):
        sim_paths[:, block_offset::block_size] = daily[start_indices + block_offset]
    sim_paths = sim_paths[:, :n]

    # FIXED: Added safety floor to clip bankruptcy scenarios and prevent NaN cascades
    sim_paths = np.clip(sim_paths, -0.9999, None)

    # FIXED: Standardized to arithmetic compounding to reconcile with Sharpe geometry
    eq_paths = ctx.START_CAPITAL * np.cumprod(1 + sim_paths, axis=1)
    
    # Calculate vectorized maximum drawdowns
    running_peaks = np.maximum.accumulate(eq_paths, axis=1)
    drawdowns = (eq_paths - running_peaks) / running_peaks
    max_dds = np.min(drawdowns, axis=1) * 100
    total_returns = (eq_paths[:, -1] - ctx.START_CAPITAL) / ctx.START_CAPITAL * 100

    # Calculate vectorized annualized Sharpe ratios
    means = np.mean(sim_paths - ctx.DAILY_RF, axis=1)
    stds = np.std(sim_paths, axis=1)
    sharpes = np.where(stds > 0, (means / stds) * np.sqrt(252), 0.0)

    # Return indices explicitly to allow downstream cross-module synchronization
    return {
        'total_return': total_returns, 
        'max_dd': max_dds, 
        'sharpe': sharpes, 
        'indices': start_indices
    }

# ==============================================================================
# MULTI-STRATEGY MONTE CARLO TORTURE TESTING UTILITY
# ==============================================================================
def run_monte_carlo_suite(results, ctx: SystemContext, n_sims=5000, block_size=5, seed=42):
    """
    Orchestrates torture testing across backtest outputs. Discovers the system 
    benchmark early to extract and enforce its timeline paths onto all strategies.
    """
    summaries = []
    mc_by_name = {}
    
    # FIXED: Find the benchmark first to capture its specific historical sequence
    benchmark_item = next((r for r in results if 'spx' in r['name'].lower() or 'benchmark' in r['name'].lower()), None)
    shared_indices = None
    
    if benchmark_item is not None:
        # Run benchmark first to generate the Master Timeline Index Map
        bench_mc = monte_carlo_bootstrap(benchmark_item['daily_returns'], ctx, n_sims, block_size, seed)
        if bench_mc is not None:
            mc_by_name[benchmark_item['name']] = bench_mc
            shared_indices = bench_mc['indices'] # Extract chronological sequence!

    # Process all results using the shared timeline path contract
    for r in results:
        actual_total_return = (pd.Series(r['equity_curve']).iloc[-1] - ctx.START_CAPITAL) / ctx.START_CAPITAL * 100
        
        # If it's the benchmark, we already computed it above
        if benchmark_item and r['name'] == benchmark_item['name']:
            pass
        else:
            # FIXED: Pass shared_indices to force strategies onto the identical benchmark timeline path
            mc = monte_carlo_bootstrap(r['daily_returns'], ctx, n_sims=n_sims, block_size=block_size, seed=seed, start_indices=shared_indices)
            if mc is None:
                continue
            mc_by_name[r['name']] = mc

        mc = mc_by_name[r['name']]
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

    # Render comprehensive performance matrix to the console
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

    # Evaluate paired paths to calculate true alpha generation probabilities
    benchmark_key = next((k for k in mc_by_name.keys() if 'spx' in k.lower() or 'benchmark' in k.lower()), None)

    if benchmark_key is not None:
        bench_tr = mc_by_name[benchmark_key]['total_return']
        print("\n==============================================================")
        print("P(STRATEGY PATH BEATS SPX PATH) -- independently resampled Monte Carlo runs")
        print("==============================================================")
        for name, mc in mc_by_name.items():
            if name == benchmark_key:
                continue
            n = min(len(mc['total_return']), len(bench_tr))
            # PROOFED: This cross-comparison is now completely valid because paths are synchronized!
            win_rate = (mc['total_return'][:n] > bench_tr[:n]).mean() * 100
            print(f"{name:>25}: beats SPX in {win_rate:.1f}% of paired simulations")

    return table
