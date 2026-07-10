# Regime-Adaptive Portfolio Infrastructure (RAPI)
A Multi-Regime Equity Backtesting & Vectorized Bootstrap Risk Framework.

This framework is engineered to audit strategy decay and performance distribution across distinct market regimes: choppy range-bound cycles, high-stress momentum expansions, and systemic liquidity crises. 

* **Current Phase:** Active Live Out-of-Sample Forward Testing (Initiated July 9, 2026)

---

## Multi-Regime Performance Sweep Results

### 1. 2007–2010 Era (Systemic Liquidity Crash / 2008 Financial Crisis)
* **Testing Window:** 2007-01-01 to 2010-01-01
* **Top Performing Architecture:** `Zhan_Production_Flawless (Unconstrained)`
* **Total Return:** +23.42% (vs SPX -27.59%) 
* **Absolute Alpha:** +51.01% return outperformance vs SPX | +1.00 Sharpe delta vs SPX
* **Max Drawdown Exposure:** -9.93% (vs SPX -56.34%)

### 2. 2015–2016 Era (Flat / Choppy Range-Bound Market)
* **Testing Window:** 2015-01-01 to 2016-06-01
* **Top Performing Architecture:** `Yin_Production_Fixed (Unconstrained)`
* **Total Return:** +16.27% (vs SPX +3.11%) 
* **Absolute Alpha:** +13.16% return outperformance vs SPX
* **Risk-Adjusted Efficiency:** Sharpe Ratio: 1.50 | Sortino Ratio: 3.45
* **Execution Metrics:** 80 Trades Completed | 63.75% Win Rate

### 3. 2023–2026 Era (Macro Bull Market Expansion)
* **Testing Window:** 2023-01-01 to 2026-07-01
* **Top Performing Architecture:** `Zhan_Production_Flawless (Manual Limits)`
* **Total Return:** +52.75% (vs SPX +60.02%)
* **Risk-Adjusted Efficiency:** Sharpe Ratio: 1.28
* **Max Drawdown Exposure:** -10.35% (vs SPX -18.90%)

---

## Risk Management & Core Algorithmic Defenses

To ensure the statistical validity of the performance arrays across changing market environments, the framework executes several core mathematical and structural safeguards:

* **Vectorized Block Bootstrap Engine:** Built an optimized, multi-dimensional matrix engine in NumPy to execute 5,000-path Monte Carlo simulations, preserving structural historical serial correlation without runtime lag.
* **Look-Ahead & Time-Travel Insulation:** Employs explicit static array pinning to `iloc[-1]` based strictly on the prior session's completed close, neutralizing intraday calculation leaks.
* **Dynamic Capital Allocation & Decrement Gates:** Re-architected flat transaction evaluation into sequential accounting loops. Available cash ledger balances and portfolio slot thresholds are updated dynamically in memory the millisecond an entry clears, preventing overlapping duplicate ticker allocation.
* **Tail-Risk Volatility Breakers:** Model C (Yin) utilizes an automated mathematical gate (`ATR14 > 2 × ATR50`) that programmatically blocks mean-reversion entries when short-term asset volatility doubles long-term baselines, shielding the system from "falling knife" cascading liquidations.

---

## Development Methodology & AI Collaboration Disclosure

The core quantitative strategies, market regime logic, asset pricing constraints, risk parameters, and mathematical architecture of this simulation framework were authored entirely by the developer. 

Advanced AI-assisted engineering utilities were actively leveraged as a collaborative productivity framework to accelerate infrastructure build-times, optimize data-processing pipelines, and refactor procedural loops. The execution environment was strictly limited to a hyper-focused, non-integrated suite:
1. **Google Search AI Mode:** Utilized for rapid software documentation parsing and API syntax verification.
2. **Microsoft Copilot Browser Ecosystem:** Utilized for preliminary script scaffolding and layout parsing.
3. **Anthropic Claude 3.5 Sonnet (Medium Effort Tuning):** Utilized as an adversarial code auditor, state-machine verifier, and multi-dimensional NumPy array vectorization engine.

All system infrastructure files and calculation loops were audited line-by-line, verified programmatically against fatal runtime `KeyErrors`, and stress-tested manually to guarantee mathematical and logical design integrity.

---

## License

This project is licensed under the terms of the **GNU General Public License v3 (GPLv3)**. Commercial utilization or closed-source redistribution of this architecture is strictly bound by copyleft disclosure mandates. See the `LICENSE` file for details.
