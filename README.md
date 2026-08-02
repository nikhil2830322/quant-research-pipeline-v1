# Nexus Engine v5

## Point-in-Time Quantitative Research Framework

Nexus Engine v5 is a research framework for testing systematic equity strategies under realistic historical constraints.

The project was built around a simple question:

> Can a small set of rule-based trading hypotheses generate repeatable risk-adjusted returns when tested without hindsight bias?

The engine combines three independent strategy models — **Yang**, **Yin**, and **Zhan** — with a regime-aware portfolio allocator, transaction cost modeling, point-in-time universe handling, and Monte Carlo stress testing.

The goal is not to predict markets. The goal is to create a controlled environment where trading hypotheses can be tested, rejected, refined, and compared.

---

# Research Philosophy

Most backtests fail because they accidentally answer the wrong question.

Common failure points include:

- Survivorship bias
- Unrealistic execution assumptions
- Excessive parameter tuning
- Ignoring portfolio-level behavior

Nexus Engine focuses on controlling those issues:

- Historical universe construction attempts to avoid selecting only today's winners.
- Indicators are calculated using only information available before each decision point.
- Entries and exits use next-day market mechanics rather than same-day assumptions.
- Transaction friction, commissions, cash drag, and gap risk are modeled explicitly.
- Strategies are evaluated using both absolute returns and benchmark-relative performance.

The system is designed as a **hypothesis testing framework**, not a prediction model.

---

# Architecture Overview

The engine is divided into several components:


Nexus Engine v5

├── Strategy Layer
│ ├── Yang (Trend Following)
│ ├── Yin (Mean Reversion)
│ └── Zhan (Volatility Expansion)
│
├── Ensemble Engine
│ ├── Market Regime Classification
│ ├── Dynamic Strategy Allocation
│ └── Portfolio Risk Controls
│
├── Execution Layer
│ ├── Slippage
│ ├── Commission
│ ├── Gap Risk Handling
│ └── Cash Drag
│
├── Universe Engine
│ └── Point-in-Time Asset Availability
│
└── Validation Layer
├── Benchmark Comparison
└── Monte Carlo Stress Testing


---

# Strategy Layer

The framework contains three independent strategy models representing different market behaviors.

---

## Yang — Trend Following

Yang attempts to capture persistent upward momentum.

### Core Signals

- Breakouts above previous highs
- Relative volume confirmation
- Moving-average trend alignment
- Volatility-adjusted stops

### Hypothesis

> Strong companies showing expanding momentum and participation may continue outperforming after confirmation.

---

## Yin — Mean Reversion

Yin targets temporary price dislocations inside longer-term uptrends.

### Core Signals

- Extreme short-term RSI weakness
- Long-term trend confirmation
- Volatility exhaustion filters
- Short holding periods

### Hypothesis

> Healthy assets experiencing temporary oversold conditions may revert toward their short-term equilibrium.

---

## Zhan — Volatility Expansion

Zhan focuses on volatility contraction followed by directional expansion.

### Core Signals

- Keltner Channel breakouts
- EMA structure
- Volume confirmation
- ATR-based risk management

### Hypothesis

> Periods of compression can create conditions where directional expansion becomes more probable.

---

# Ensemble Engine

The ensemble layer combines the three strategies rather than assuming one approach works in every environment.

The allocator classifies the market into three regimes:

| Regime | Allocation Preference | Reasoning |
|---|---|---|
| Bull | Yang weighted highest, Zhan secondary, Yin disabled | Strong trends favor momentum continuation |
| Bear | Yin prioritized, Yang disabled | Avoids chasing broad market weakness |
| Chop | Zhan prioritized, Yin secondary | Rangebound markets require selective setups |

These allocations are intentionally asymmetric.

The system does not force every strategy to remain active because different market environments reward different behaviors.

The design decision was to allow strategies to become inactive when their expected edge is weakest rather than forcing constant participation.

---

# Risk Management

Risk controls operate at the portfolio level rather than only at individual trade level.

The framework includes:

- ATR-based position sizing
- Maximum portfolio positions
- Sector concentration limits
- Portfolio drawdown scaling
- Volatility-based exposure adjustment
- Cash drag modeling
- Entry and exit friction
- Overnight gap handling

Position size is determined by the more conservative constraint between:

1. Maximum loss allowed per trade
2. Maximum portfolio allocation

This prevents oversized positions from forming simply because volatility temporarily decreases.

---

# Point-in-Time Universe Construction

A major source of historical backtest error is survivorship bias.

For example, testing only today's largest companies ignores companies that previously existed in the index but later disappeared.

Nexus Engine attempts to reduce this issue through historical constituent tracking.

The universe system supports:

- Historical membership changes
- IPO availability checks
- Dynamic asset availability
- Removal of unavailable securities from earlier periods

Current testing uses a controlled historical basket and does **not** claim to represent a complete reconstruction of every historical S&P 500 constituent.

---

# Execution Model

The engine uses next-day execution assumptions.

## Trade Flow

1. Signal generated using previous closing data.
2. Entry occurs using the following day's opening price.
3. Exit logic evaluates previous information.
4. Execution occurs using available market prices.

Execution friction includes:

- Entry slippage
- Exit slippage
- Commission costs
- Gap-down stop behavior

This prevents same-bar execution advantages that are unavailable in live trading.

---

# Monte Carlo Stress Testing

Historical backtests only represent one possible market path.

Nexus Engine applies stationary block bootstrap Monte Carlo simulations to evaluate robustness.

## Simulation Parameters

| Parameter | Value |
|---|---:|
| Simulations | 5,000 |
| Block Size | 5 trading days |
| Sampling Method | Stationary block bootstrap |

## Metrics Evaluated

The simulations measure:

- Return distribution
- Drawdown distribution
- Sharpe distribution
- Probability of positive outcomes
- Probability of outperforming benchmark paths

Monte Carlo results are treated as robustness analysis, not proof of future performance.

---

# Example Validation Results

## Testing Period

**2007-01-01 → 2010-01-01**

Market conditions included:

- The 2008 financial crisis
- A major equity drawdown environment
- Comparison against S&P 500 buy-and-hold

| Model | Return | Max Drawdown | Sharpe | Trades |
|---|---:|---:|---:|---:|
| Yang (Manual Limits) | 25.23% | -6.48% | 0.63 | 69 |
| Zhan (Manual Limits) | 13.95% | -6.96% | 0.21 | 74 |
| Yin (Manual Limits) | 23.87% | -9.70% | 0.65 | 260 |
| Ensemble | 18.17% | -7.97% | 0.39 | 95 |
| S&P 500 Buy & Hold | -27.59% | -56.34% | -0.40 | — |

These results should not be interpreted as guarantees of future returns.

They represent a single historical sample used to evaluate whether the underlying hypotheses behaved as intended.

---

# Current Limitations

This project remains a research system.

Known limitations:

- The current point-in-time universe is a controlled dataset rather than a complete historical reconstruction of all S&P 500 constituents.
- Parameter selection has not yet been fully separated into training and validation periods.
- Additional market environments are required before stronger conclusions can be drawn.
- Real-world execution introduces additional constraints including liquidity, spreads, and market impact.

---

# Future Development

Planned extensions:

- Full historical S&P 500 constituent database integration
- Walk-forward optimization
- Out-of-sample validation periods
- More realistic liquidity constraints
- Portfolio correlation controls
- Additional asset classes
- Live paper-trading integration

---

# Project Summary

Nexus Engine v5 is an attempt to bridge the gap between simple trading scripts and systematic research.

The core principle is:

> Create explicit hypotheses, test them honestly, measure failure modes, and improve the system based on evidence rather than assumptions.

| Component                   | Ownership           |
| --------------------------- | ------------------- |
| Research questions          | Student-designed    |
| Financial hypotheses        | Student-designed    |
| Risk framework              | Student-designed    |
| Validation methodology      | Student-designed    |
| Implementation acceleration | AI-assisted         |
| Future reconstruction       | Independent rebuild |


Inludes: 14 generations of a self-taught, modular quantitative research pipeline (NEXUS v5) including vectorized path-synchronized Monte Carlo simulations.

"Note on Development Workflow: Due to intermittent internet and hardware constraints, core iterations were engineered and stress-tested within Google Colab environments before being committed as complete versions to this repository."
