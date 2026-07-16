import numpy as np
import pandas as pd

# Core Configuration, Ingestion, and Global Settings Imports
from config_and_data import SystemContext, load_data, build_indicators, tickers

# Model Strategy Modules Imports
from model_yang import run_yang
from model_yin import run_yin
from model_zhan import run_zhan
from ensemble_engine import run_ensemble_suite

# Performance Evaluation and Statistical Torture Test Imports
from evaluation_and_metrics import compare_models, run_benchmark
from monte_carlo_engine import run_monte_carlo_suite

# ==============================================================================
# CONSTRAINT SWEEP - Manual vs Unconstrained
# ==============================================================================
def run_full_suite(ind, data, sp500, sp500_sma200, sp500_sma50,
                    max_total_positions, max_alloc_fraction, max_sector_positions,
                    label="", base_ctx=None):
    """
    Runs Yang/Zhan/Yin with custom isolated position-limit parameters.
    Instantiates sandboxed contexts to eliminate namespace leakage down the pipeline.
    """
    if base_ctx is None:
        base_ctx = SystemContext()
        
    # Instantiate custom sandboxed configuration objects for the target run sweep
    run_ctx = SystemContext(
        start_capital=base_ctx.START_CAPITAL,
        risk_per_trade=base_ctx.RISK_PER_TRADE,
        max_sector_positions=max_sector_positions,
        max_total_positions=max_total_positions,
        max_alloc_fraction=max_alloc_fraction,
        target_vol=base_ctx.TARGET_VOL,
        start_date=base_ctx.START_DATE,
        end_date=base_ctx.END_DATE
    )

    suffix = f" {label}" if label else ""
    
    # Execute strategy runs independently via explicit context parameters
    yang = run_yang(ind, data, sp500, sp500_sma200, sp500_sma50, run_ctx)
    zhan = run_zhan(ind, data, sp500, sp500_sma200, sp500_sma50, run_ctx)
    yin = run_yin(ind, data, sp500, sp500_sma200, sp500_sma50, run_ctx)
    
    yang['name'] += suffix
    zhan['name'] += suffix
    yin['name'] += suffix

    return yang, zhan, yin

# ==============================================================================
# MASTER SYSTEMS EXECUTION GATEWAY
# ==============================================================================
if __name__ == "__main__":
    # Initialize the baseline environment context
    master_ctx = SystemContext()
    
    # Phase 1: Ingest Data Elements Safely via explicit context dates
    data = load_data(master_ctx)
    
    # Phase 2: Compute Rolling Metric Transformations
    ind, sp500, sp500_sma200, sp500_sma50 = build_indicators(data)
    dates = sp500.index[200:]

    print("\n### RUN 1: Manual-Tracking Limits (original) ###")
    print("    MAX_TOTAL_POSITIONS=6, MAX_ALLOC_FRACTION=1/6, MAX_SECTOR_POSITIONS=2")
    yang_manual, zhan_manual, yin_manual = run_full_suite(
        ind, data, sp500, sp500_sma200, sp500_sma50,
        max_total_positions=6, max_alloc_fraction=1/6, max_sector_positions=2,
        label="(Manual Limits)",
        base_ctx=master_ctx
    )

    print("\n### RUN 2: Unconstrained (no manual-tracking caps) ###")
    print(f"    MAX_TOTAL_POSITIONS={len(tickers)}, MAX_ALLOC_FRACTION=1.0, "
          f"MAX_SECTOR_POSITIONS={len(tickers)}")
    yang_free, zhan_free, yin_free = run_full_suite(
        ind, data, sp500, sp500_sma200, sp500_sma50,
        max_total_positions=len(tickers), max_alloc_fraction=1.0, max_sector_positions=len(tickers),
        label="(Unconstrained)",
        base_ctx=master_ctx
    )

    print("\n### RUN 3: Ensemble (dynamic Yin/Yang/Zhan blend) ###")
    ensemble_results = run_ensemble_suite(
        ind, data, sp500, sp500_sma200, sp500_sma50,
        config_params={'max_total_positions': 8, 'max_alloc_fraction': 1/6,
                        'max_sector_positions': 2, 'target_vol': 0.10}
    )

    print("\nProcessing SPX Buy & Hold benchmark...")
    benchmark_results = run_benchmark(sp500, dates, master_ctx)

    # Phase 3: Compile Complete Performance Tracking Matrix
    all_results = [
        yang_manual, zhan_manual, yin_manual,
        yang_free, zhan_free, yin_free,
        ensemble_results,
        benchmark_results
    ]

    # Display clean comparative performance statistics
    compare_models(all_results, master_ctx, SHOW_CHARTS=False)

    print("\nRunning Monte Carlo block-bootstrap analysis...")
    run_monte_carlo_suite(all_results, master_ctx, n_sims=5000, block_size=5, seed=42)


