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
# CONSTRAINT SWEEP - Manual vs Unconstrained (RE-ALIGNED)
# ==============================================================================
def run_full_suite(ind, data, sp500, sp500_sma200, sp500_sma50,
                    max_total_positions, max_alloc_fraction, max_sector_positions,
                    label="", base_ctx=None, universe_history=None):
    """
    Runs Yang/Zhan/Yin with custom isolated position-limit parameters.
    """
    if base_ctx is None:
        base_ctx = SystemContext()
        
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
    
    yang = run_yang(ind, data, sp500, sp500_sma200, sp500_sma50, run_ctx)
    zhan = run_zhan(ind, data, sp500, sp500_sma200, sp500_sma50, run_ctx)
    yin = run_yin(ind, data, sp500, sp500_sma200, sp500_sma50, run_ctx)
    
    yang['name'] += suffix
    zhan['name'] += suffix
    yin['name'] += suffix

    return yang, zhan, yin


# ==============================================================================
# MASTER SYSTEMS EXECUTION GATEWAY (VERSION 4 FULLY UNLOCKED)
# ==============================================================================
if __name__ == "__main__":
    # Initialize the baseline environment context
    master_ctx = SystemContext()
    
    # Phase 0: Reconstruct Point-in-Time Matrix Timeline (PIT Refactor)
    print("Compiling Point-in-Time S&P 500 asset history from Farrell's local file...")
    raw_universe_history = get_historical_spx_constituents(start_date=master_ctx.START_DATE)
    
    # Clean and pre-sanitize the master tickers container list for safety
    clean_core_tickers = {str(t).strip().upper().replace('.', '-') for t in tickers}
    
    # --------------------------------------------------------------------------
    # PRODUCTION INTERSECTION FILTER PASS
    # Intersects Farrell's data while stripping all trailing spaces and formatting artifacts.
    # This limits the download strictly to assets inside your core 31-ticker portfolio.
    # --------------------------------------------------------------------------
    all_historical_tickers = set()
    sanitized_universe_history = {}
    
    for date_str, ticker_list in raw_universe_history.items():
        sanitized_day_list = []
        for t in ticker_list:
            clean_t = str(t).strip().upper().replace('.', '-')
            # Only keep the asset if it is a member of your core portfolio list
            if clean_t in clean_core_tickers:
                sanitized_day_list.append(clean_t)
                all_historical_tickers.add(clean_t)
        
        # Populate history map chronologically
        sanitized_universe_history[date_str] = sanitized_day_list
        
    all_historical_tickers = sorted(list(all_historical_tickers))
    print(f"Dynamic matrix initialized. Total active point-in-time pool size: {len(all_historical_tickers)} tickers.")

    # Phase 1: Ingest Data Elements safely for the valid historical candidates
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
    # In this version, len(all_historical_tickers) dynamically evaluates to ~26 assets during the 2007 test
    print(f"    MAX_TOTAL_POSITIONS={len(all_historical_tickers)}, MAX_ALLOC_FRACTION=1.0, "
          f"MAX_SECTOR_POSITIONS={len(all_historical_tickers)}")
    yang_free, zhan_free, yin_free = run_full_suite(
        ind, data, sp500, sp500_sma200, sp500_sma50,
        max_total_positions=len(all_historical_tickers), max_alloc_fraction=1.0, max_sector_positions=len(all_historical_tickers),
        label="(Unconstrained)",
        base_ctx=master_ctx
    )

    print("\n### RUN 3: Ensemble (dynamic Yin/Yang/Zhan blend) ###")
    try:
        ensemble_results = run_ensemble_suite(
            ind, data, sp500, sp500_sma200, sp500_sma50,
            config_params={'max_total_positions': 8, 'max_alloc_fraction': 1/6,
                            'max_sector_positions': 2, 'target_vol': 0.10},
            universe_history=sanitized_universe_history  # Pass sanitized history map
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
