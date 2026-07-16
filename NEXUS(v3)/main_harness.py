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

# Point-in-Time Data Engineering Imports (Version 3 Refactor)
from universe_factory import get_historical_spx_constituents

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
    
    # Phase 0: Reconstruct Point-in-Time Matrix Timeline (PIT Refactor)
    print("Compiling Point-in-Time S&P 500 asset history...")
    spx_universe_history = get_historical_spx_constituents(start_date=master_ctx.START_DATE)
    
    # Extract unique global tickers that ever appeared in this regime window
    all_historical_tickers = set()
    for date_str, ticker_list in spx_universe_history.items():
        all_historical_tickers.update(ticker_list)
    all_historical_tickers = list(all_historical_tickers)
    print(f"Dynamic matrix initialized. Total historical pool size: {len(all_historical_tickers)} tickers.")

    # Phase 1: Ingest Data Elements safely for ALL historical candidates
    # Pass the full PIT universe array down into your data module
    data = load_data(master_ctx, target_universe=all_historical_tickers)
    
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
    # Dynamically match max configuration bounds to the length of your live running matrix
    print(f"    MAX_TOTAL_POSITIONS={len(all_historical_tickers)}, MAX_ALLOC_FRACTION=1.0, "
          f"MAX_SECTOR_POSITIONS={len(all_historical_tickers)}")
    yang_free, zhan_free, yin_free = run_full_suite(
        ind, data, sp500, sp500_sma200, sp500_sma50,
        max_total_positions=len(all_historical_tickers), max_alloc_fraction=1.0, max_sector_positions=len(all_historical_tickers),
        label="(Unconstrained)",
        base_ctx=master_ctx
    )

    print("\n### RUN 3: Ensemble (dynamic Yin/Yang/Zhan blend) ###")
    # FIXED: Adaptive try/except block passes down universe maps gracefully even if your 
    # ensemble engine code file signature hasn't been recompiled yet
    try:
        ensemble_results = run_ensemble_suite(
            ind, data, sp500, sp500_sma200, sp500_sma50,
            config_params={'max_total_positions': 8, 'max_alloc_fraction': 1/6,
                            'max_sector_positions': 2, 'target_vol': 0.10},
            universe_history=spx_universe_history
        )
    except TypeError:
        print("WARNING: ensemble_engine signature mismatch detected. Executing fallback route container parameters...")
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
