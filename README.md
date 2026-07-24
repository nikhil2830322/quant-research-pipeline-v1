# Regime-Adaptive Portfolio Infrastructure (RAPI)

**A Multi-Regime Equity Backtesting & Vectorized Bootstrap Risk Framework**

> **Current Phase:** Active Live Out-of-Sample Forward Testing *(Initiated July 9, 2026)*

**Contact**
- 📧 nikhilsai28303@gmail.com
- 📧 nikhilsai28303@outlook.com

---

# 📊 Multi-Regime Performance Sweep Results

## 1. 2007–2010 Era (Systemic Liquidity Crash / 2008 Financial Crisis)

**Testing Window:** `2007-01-01 → 2010-01-01`

### Top Performing Architecture
**Zhan_Production_Flawless (Unconstrained)**

| Metric | Result |
|---------|--------|
| Total Return | **+37.12%** *(SPX: -27.59%)* |
| Absolute Alpha | **+64.71%** return outperformance vs. SPX |
| Sharpe Delta vs. SPX | **+1.39** |
| Max Drawdown | **-9.70%** *(SPX: -56.34%)* |
| Sharpe Ratio | **0.99** |
| Sortino Ratio | **1.50** |

---

## 2. 2015–2016 Era (Flat / Choppy Range-Bound Market)

**Testing Window:** `2015-01-01 → 2016-06-01`

### Top Performing Architecture
**Yin_Production_Fixed (Unconstrained)**

| Metric | Result |
|---------|--------|
| Total Return | **+16.27%** *(SPX: +3.11%)* |
| Absolute Alpha | **+13.16%** return outperformance vs. SPX |
| Sharpe Ratio | **1.50** |
| Sortino Ratio | **3.45** |
| Trades Completed | **80** |
| Win Rate | **63.75%** |

---

## 3. 2023–2026 Era (Macro Bull Market Expansion)

**Testing Window:** `2023-01-01 → 2026-07-01`

### Top Performing Architecture
**Zhan_Production_Flawless (Manual Limits)**

| Metric | Result |
|---------|--------|
| Total Return | **+52.75%** *(SPX: +60.02%)* |
| Sharpe Ratio | **1.28** |
| Max Drawdown | **-10.35%** *(SPX: -18.90%)* |

---

# 🛡️ Risk Management & Core Algorithmic Defenses

To ensure the statistical validity of the performance arrays across changing market environments, the framework executes several core mathematical and structural safeguards.

## Vectorized Block Bootstrap Engine

- Optimized multi-dimensional NumPy matrix engine
- Executes **5,000-path Monte Carlo simulations**
- Preserves historical serial correlation
- Fully vectorized implementation with minimal runtime overhead

## Look-Ahead & Time-Travel Insulation

- Explicit static array pinning
- Uses strictly `iloc[-1]` from the prior completed session
- Eliminates intraday look-ahead bias
- Handles unavailable historical listings and pre-IPO securities
- Safely catches `YFPricesMissingError` (e.g., TSLA during 2007 testing)

## Dynamic Capital Allocation & Decrement Gates

- Sequential accounting engine replaces flat transaction evaluation
- Cash ledger updates immediately after every executed trade
- Portfolio slot availability is updated dynamically in memory
- Prevents duplicate ticker allocations and overlapping position sizing

## Tail-Risk Volatility Breakers

Model **Yin (Model C)** automatically blocks mean-reversion entries whenever:

```
ATR₁₄ > 2 × ATR₅₀
```

This mathematical gate prevents entries during volatility explosions and helps protect against cascading "falling knife" drawdowns.

---

# 🛠️ Core Backtesting Architecture: The NEXUS Engines

The simulation engine is divided into two execution tracks depending on available compute resources and desired asset universe.

## 📐 Pipeline Data Flow

```text
universe_factory (PIT Core)
         │
         ▼
 config_and_data
         │
         ▼
risk_and_friction
         │
         ▼
  model_modules
         │
         ▼
 ensemble_engine
         │
         ▼
evaluation_and_metrics
         │
         ▼
 monte_carlo_engine
```

---

# 📁 Institutional Module Breakdown

| Module | Description |
|---------|-------------|
| `main_harness.py` | Coordinates simulation epochs, portfolio limits, and initializes the bootstrap stress-testing engine. |
| `universe_factory.py` | Point-in-Time (PIT) universe builder using Farrell Aultman's offline database. |
| `config_and_data.py` | Central configuration registry and SystemContext initialization (Upgraded v5 Production Core). |
| `risk_and_friction.py` | Commission modeling, non-linear volatility targeting, and multi-position sector caps (Upgraded v5 Production Core). |
| `model_yang.py` | Breakout strategy implementation (Upgraded v5 Production Core). |
| `model_yin.py` | Mean-reversion strategy implementation. |
| `model_zhan.py` | Volatility squeeze strategy implementation. |
| `ensemble_engine.py` | Portfolio manager combining all strategy outputs and handling intra-day capital drainage (`sizing_equity -= total_cost`). |
| `evaluation_and_metrics.py` | Performance statistics and institutional risk metrics. |
| `monte_carlo_engine.py` | 5,000-path vectorized block bootstrap simulation engine. |

---

# ⚡ Choose Your Execution Track

## 🟢 NEXUS v5 Production Baseline (Recommended)

### Upgraded Systems

- Optimized Risk Module
- Optimized Yang Module
- Optimized Configuration Module

### Target Scope

Curated Point-in-Time institutional portfolio of approximately **26–31 blue-chip monopoly businesses**.

### Characteristics

| Feature | Value |
|---------|-------|
| Runtime | ~20 seconds |
| Compute | Google Colab Free CPU |
| Universe | Curated PIT Portfolio |
| Survivorship Bias | Eliminated through Institutional Intersection Filter (2007–2010) |

---

## 🔴 NEXUS v4 Experimental High-Compute Alpha Track

### Characteristics

| Feature | Value |
|---------|-------|
| Universe | Full historical S&P 500 |
| Asset Count | 929+ unique tickers |
| Runtime | 35+ minutes |
| Compute | High-RAM / HPC |
| Notes | Full Farrell database sweep with significant Pandas fragmentation |

---

# 🎲 Stochastic Path Seeding

## Verified Metrics (2007–2010)

**5,000-run Vectorized Block Bootstrap (5-Day Blocks)**

| Strategy | Median Return | Median Sharpe | Probability Strategy Outperforms SPX |
|-----------|--------------|---------------|--------------------------------------|
| Zhan (Unconstrained) | **+39.68%** | **1.08** | **94.4%** |
| Yin (Unconstrained) | **+27.70%** | **0.56** | **90.3%** |
| Yang (Manual Limits) | **+26.63%** | **0.70** | **90.9%** |
| Ensemble Blend | **+20.10%** | **0.48** | **89.0%** |
| SPX Buy & Hold | **-25.81%** | **-0.37** | Baseline |

---

# 🧬 Development Methodology & AI Collaboration Disclosure

The strategies, architecture, research direction, and implementation are authored by the developer. AI tools were used solely as engineering assistants for optimization, documentation, and code auditing.

| Tool | Primary Use |
|------|-------------|
| Google Search AI Mode | Documentation and syntax reference |
| Microsoft Copilot | Script scaffolding |
| Anthropic Claude 4.6 Sonnet | Auditing and NumPy vectorization |

All systems were subsequently validated for runtime safety and implementation correctness.

---

# 📜 License

This project is licensed under the **GNU General Public License v3 (GPLv3)**.
