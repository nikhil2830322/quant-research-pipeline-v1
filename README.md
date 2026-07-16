# Regime-Adaptive Portfolio Infrastructure (RAPI)
### A Multi-Regime Equity Backtesting & Vectorized Bootstrap Risk Framework
*Current Phase: Active Live Out-of-Sample Forward Testing (Initiated July 9, 2026)*

---
Contact Info: nikhilsai28303@gmail.com, nikhilsai28303@outlook.com,
## 📊 Multi-Regime Performance Sweep Results

### 1. 2007–2010 Era (Systemic Liquidity Crash / 2008 Financial Crisis)
*   **Testing Window:** 2007-01-01 to 2010-01-01
*   **Top Performing Architecture:** `Zhan_Production_Flawless (Unconstrained)`
*   **Total Return:** +23.42% (vs SPX -27.59%) 
*   **Absolute Alpha:** +51.01% return outperformance vs SPX | +1.00 Sharpe delta vs SPX
*   **Max Drawdown Exposure:** -9.93% (vs SPX -56.34%)

### 2. 2015–2016 Era (Flat / Choppy Range-Bound Market)
*   **Testing Window:** 2015-01-01 to 2016-06-01
*   **Top Performing Architecture:** `Yin_Production_Fixed (Unconstrained)`
*   **Total Return:** +16.27% (vs SPX +3.11%) 
*   **Absolute Alpha:** +13.16% return outperformance vs SPX
*   **Risk-Adjusted Efficiency:** Sharpe Ratio: 1.50 | Sortino Ratio: 3.45
*   **Execution Metrics:** 80 Trades Completed | 63.75% Win Rate

### 3. 2023–2026 Era (Macro Bull Market Expansion)
*   **Testing Window:** 2023-01-01 to 2026-07-01
*   **Top Performing Architecture:** `Zhan_Production_Flawless (Manual Limits)`
*   **Total Return:** +52.75% (vs SPX +60.02%)
*   **Risk-Adjusted Efficiency:** Sharpe Ratio: 1.28
*   **Max Drawdown Exposure:** -10.35% (vs SPX -18.90%)

---

## 🛡️ Risk Management & Core Algorithmic Defenses

To ensure the statistical validity of the performance arrays across changing market environments, the framework executes several core mathematical and structural safeguards:

*   **Vectorized Block Bootstrap Engine**: Built an optimized, multi-dimensional matrix engine in NumPy to execute 5,000-path Monte Carlo simulations, preserving structural historical serial correlation without runtime lag.
*   **Look-Ahead & Time-Travel Insulation**: Employs explicit static array pinning to `iloc[-1]` based strictly on the prior session's completed close, neutralizing intraday calculation leaks.
*   **Dynamic Capital Allocation & Decrement Gates**: Re-architected flat transaction evaluation into sequential accounting loops. Available cash ledger balances and portfolio slot thresholds are updated dynamically in memory the millisecond an entry clears, preventing overlapping duplicate ticker allocation.
*   **Tail-Risk Volatility Breakers**: Model C (Yin) utilizes an automated mathematical gate (`ATR14 > 2 × ATR50`) that programmatically blocks mean-reversion entries when short-term asset volatility doubles long-term baselines, shielding the system from "falling knife" cascading liquidations.

---

## 🛠️ CORE BACKTESTING ARCHITECTURE: THE NEXUS ENGINES

The simulation engine is split into two distinct execution tracks depending on available local computing power and target asset pool constraints.

### 📐 Pipeline Data Flow
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

### 📁 Institutional Module Breakdown
*   **`main_harness.py`**: Coordinates simulation epochs, handles portfolio limits, and boots up the bootstrap stress desk.
*   **`universe_factory.py`**: Point-in-Time (PIT) universe builder. Reads Farrell Aultman's database completely offline.
*   **`config_and_data.py`**: Manages central setting registries and initializes `SystemContext` configurations.
*   **`risk_and_friction.py`**: Handles commission friction, non-linear volatility targeting, and multi-position sector caps.
*   **`model_yang.py`** / **`model_yin.py`** / **`model_zhan.py`**: Underlying breakout, mean-reversion, and squeeze strategy modules.
*   **`ensemble_engine.py`**: Core portfolio manager. Combines all rules and manages intra-day drainage waterfalls (`sizing_equity -= total_cost`).
*   **`evaluation_and_metrics.py`** & **`monte_carlo_engine.py`**: Compiles risk ratios and runs the 5,000-path bootstrap simulation.

---

## ⚡ CHOOSE YOUR EXECUTION TRACK (v3 Production vs. v4 Experimental)

### 🟢 TRACK 1: NEXUS v3 Production Baseline (Recommended)
*   **Target Scope:** Curated Point-in-Time Operational Portfolio (~26-31 Core Blue-Chip Monopolies).
*   **Execution Time:** **~ 2 mins** for a full 3-year historical regime backtest.
*   **Compute Required:** Low (Runs flawlessly on standard Google Colab free-tier CPU instances or baseline laptops).
*   **Behavior:** Uses the **Institutional Intersection Filter Pass** at the harness perimeter gate. It cross-references Farrell's historical changes file rows against your operational core portfolio upfront. This completely erases survivorship bias (dropping `META` and `TSLA` dynamically from 2007 runs), but keeps the active dataframe indicators small, fast, and light.

### 🔴 TRACK 2: NEXUS v4 Experimental High-Compute Alpha Track
*   **Target Scope:** Unconstrained Full-Scale Historical S&P 500 Index (**929+ Unique Active Tickers** simultaneously).
*   **Execution Time:** **35+ Minutes** (High risk of CPU thrashing, memory fragmentation, or instance timeout).
*   **Compute Required:** **Extreme High-Performance Computing (HPC)** / Multi-Core Cloud Threading / High-RAM runtime environments.
*   **Behavior:** Drops all intersection perimeter restrictions entirely. It loops through Farrell's raw text cells, splits the comma-separated strings, and passes **hundreds of active companies** into the loop pipeline concurrently. 
*   **⚠️ WARNING:** Because it scans 929 assets across 756 individual trading days, it triggers close to a million nested dataframe `.loc` indexing calls, causing a severe **Pandas Copy-on-Write and Index Fragmentation Penalty** inside `BlockManager.get_slice()`. *Do not execute Track 2 on consumer hardware or low-tier cloud VMs.*

---

## 🧬 Development Methodology & AI Collaboration Disclosure

The core quantitative strategies, market regime logic, asset pricing constraints, risk parameters, and mathematical architecture of this simulation framework were authored entirely by the developer. 

Advanced AI-assisted engineering utilities were actively leveraged as a collaborative productivity framework to accelerate infrastructure build-times, optimize data-processing pipelines, and refactor procedural loops. The execution environment was strictly limited to a hyper-focused, non-integrated suite:
1.  **Google Search AI Mode:** Utilized for rapid software documentation parsing and API syntax verification.
2.  **Microsoft Copilot Browser Ecosystem:** Utilized for preliminary script scaffolding and layout parsing.
3.  **Anthropic Claude 4.6 Sonnet (Low Effort Tuning):** Utilized as an adversarial code auditor, state-machine verifier, and multi-dimensional NumPy array vectorization engine.

All system infrastructure files and calculation loops were audited line-by-line, verified programmatically against fatal runtime `KeyErrors`, and stress-tested manually to guarantee mathematical and logical design integrity.

---

## 📜 License

This project is licensed under the terms of the **GNU General Public License v3 (GPLv3)**. Commercial utilization or closed-source redistribution of this architecture is strictly bound by copyleft disclosure mandates. See the `LICENSE` file for details.
