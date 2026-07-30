# NEXUS Engine v5
## A Point-in-Time Adaptive Portfolio Research Framework

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![Research](https://img.shields.io/badge/Project-Quantitative%20Research-green.svg)
![Backtesting](https://img.shields.io/badge/Framework-Point--in--Time%20Backtesting-orange.svg)

---

# Overview

NEXUS Engine v5 is a hypothesis-driven quantitative portfolio research framework designed to test whether multiple market behavior models can be combined into a regime-adaptive allocation system.

Rather than optimizing a single trading strategy, NEXUS separates the investment process into independent research components:

- **Signal hypotheses**
- **Market regime classification**
- **Portfolio allocation**
- **Risk management**
- **Transaction friction modeling**
- **Robustness testing**

The objective is not simply maximizing historical returns. The objective is determining whether different market behaviors can complement each other while controlling downside risk across different environments.

---

# Research Philosophy

Markets exhibit different behavioral regimes.

A single strategy may perform well in one environment and fail in another:

| Market Environment | Expected Behavior |
|---|---|
| Strong trends | Momentum continuation |
| Volatility expansion | Breakout continuation |
| Temporary dislocations | Mean reversion |
| Crisis environments | Capital preservation |

NEXUS models these behaviors independently through three rule-based research modules:


            Market Data
                |
                v
    +-------------------------+
    |  Point-In-Time Engine   |
    +-------------------------+
                |
                v
    +-------------------------+
    |  Signal Hypothesis Layer |
    +-------------------------+
      |          |          |
      v          v          v

    Yang       Zhan       Yin
   Trend    Volatility   Mean
  Momentum  Expansion  Reversion

      \          |          /
       \         |         /
        v        v        v

    Regime Adaptive Allocator

                |
                v

      Portfolio Risk Engine

                |
                v

      Monte Carlo Validation


---

# Core Architecture

## 1. Point-In-Time Universe Construction

NEXUS avoids using future information when constructing historical universes.

The universe layer attempts to reproduce which assets were available during each historical period.

Features:

- Historical constituent tracking
- Dynamic asset availability
- IPO availability checks
- Missing-data protection
- Historical survivorship control

Current validation uses a curated point-in-time tracked equity universe.

Future versions will expand validation to the complete historical S&P 500 constituent database.

---

# Strategy Hypotheses

## Yang — Trend Momentum Model

**Hypothesis:**

> Persistent price trends combined with volume confirmation can generate positive risk-adjusted returns.

Characteristics:

- Moving-average trend filter
- Breakout detection
- Volume confirmation
- ATR-based risk management
- Trailing stop exits

Primary environment:

- Bull markets
- Persistent directional trends

---

## Zhan — Volatility Expansion Model

**Hypothesis:**

> Periods of volatility compression followed by expansion can identify asymmetric breakout opportunities.

Characteristics:

- Keltner channel expansion
- EMA breakout detection
- Volume regime analysis
- Consolidation filtering
- ATR trailing stops

Primary environment:

- Momentum transitions
- Volatility expansion phases

---

## Yin — Mean Reversion Model

**Hypothesis:**

> Short-term oversold conditions can recover when long-term structural trends remain intact.

Characteristics:

- RSI extreme detection
- Long-term trend filter
- Volatility protection
- Short holding periods
- Risk-defined entries

Primary environment:

- Pullbacks inside long-term uptrends
- Market dislocations

---

# Ensemble Allocation Engine

NEXUS does not combine strategies through static weights.

Instead, allocation changes dynamically according to market regime.

## Regime Classification

The engine classifies conditions into:

- BULL
- CHOP
- BEAR

Based on:

- Price relative to moving averages
- Market momentum
- Realized volatility expansion

---

# Dynamic Strategy Allocation

Example allocation logic:

## Bull Regime


Yang: 70%
Zhan: 30%
Yin: 0%


Focus:

- Trend participation
- Breakout continuation

---

## Bear Regime


Yang: 0%
Zhan: 40%
Yin: 60%


Focus:

- Defensive opportunities
- Mean-reversion recovery

---

## Chop Regime


Yang: 15%
Zhan: 55%
Yin: 30%


Focus:

- Balanced opportunity capture

---

# Risk Management Framework

NEXUS includes multiple portfolio-level controls.

## Position Sizing

Risk-based sizing:

- Maximum allocation limits
- ATR-based stop distances
- Risk-per-trade constraints

---

## Portfolio Controls

Includes:

- Sector concentration limits
- Maximum position counts
- Cash drag modeling
- Transaction friction
- Slippage assumptions

---

## Drawdown Guard

The portfolio dynamically reduces exposure during significant losses.

Example:


Normal conditions:
100% sizing

Moderate drawdown:
85% sizing

Large drawdown:
50% sizing

Extreme drawdown:
Capital preservation mode


---

# Transaction Modeling

The framework includes:

- Entry slippage
- Exit slippage
- Commission costs
- Overnight gap risk

Stops are not assumed to execute perfectly.

If a security gaps below a stop level, execution occurs at the available market price.

---

# Validation Framework

NEXUS evaluates performance using multiple layers.

## Performance Metrics

Calculated metrics:

- Total return
- Maximum drawdown
- Sharpe ratio
- Sortino ratio
- Profit factor
- Win rate
- Trade statistics

---

## Monte Carlo Robustness Testing

The engine performs:

- 5,000 bootstrap simulations
- Stationary block resampling
- Return distribution analysis
- Drawdown stress testing
- Paired benchmark comparison

The objective:

> Determine whether observed performance survives alternate market path sequences.

---

# Example Validation Results

## Test Period


2007-01-01 to 2010-01-01


This period includes:

- 2008 Financial Crisis
- Extreme volatility expansion
- Severe market drawdown

---

## Benchmark


S&P 500 Buy & Hold

Return:
-27.59%

Maximum Drawdown:
-56.34%

Sharpe:
-0.40


---

## Strategy Results

| Model | Return | Max DD | Sharpe |
|-|-:|-:|-:|
| Yang Manual | +25.23% | -6.48% | 0.63 |
| Zhan Manual | +13.95% | -6.96% | 0.21 |
| Yin Manual | +23.87% | -9.70% | 0.65 |
| Ensemble | +18.17% | -7.97% | 0.39 |

---

# Monte Carlo Results

Example:

| Model | Profitable Simulations |
|-|-:|
| Yang Manual | 95.16% |
| Yin Manual | 94.42% |
| Zhan Unconstrained | 95.14% |
| Ensemble | 91.70% |

Paired path testing:


Probability of beating SPX:

Yang Manual:
90.9%

Ensemble:
89.0%


---

# Important Limitations

NEXUS is a research framework, not a production trading system.

Current limitations:

## Universe Coverage

Current testing uses a curated point-in-time tracked equity universe.

A full historical S&P 500 constituent reconstruction remains future work.

---

## Historical Validation

Current results represent limited historical periods.

Additional testing required:

- Multiple market cycles
- Different asset universes
- Walk-forward validation
- Out-of-sample periods

---

## Parameter Selection

Parameters are hypothesis-driven.

No machine-learning optimization or parameter mining was performed.

Future versions should evaluate:

- Parameter sensitivity
- Robustness surfaces
- Walk-forward optimization

---

# Project Goals

Future development:

## Research Improvements

- Full historical S&P 500 reconstruction
- Multi-period validation
- Automated experiment tracking
- Strategy attribution analysis

## Portfolio Improvements

- More asset classes
- Dynamic correlation modeling
- Volatility targeting improvements
- Exposure attribution

---

# Key Design Principles

NEXUS follows several principles:

1. **Avoid future information**
2. **Separate hypothesis from allocation**
3. **Model realistic execution**
4. **Measure robustness, not only returns**
5. **Prefer explainable systems over opaque optimization**

---

# Conclusion

NEXUS Engine v5 demonstrates a framework for combining multiple market hypotheses into an adaptive portfolio research system.

The project focuses on a central quantitative research question:

> Can independently designed market behavior models be dynamically allocated to produce more robust outcomes than isolated strategies?

The answer requires continued validation, but the framework provides the infrastructure needed to test that question rigorously.

---

# Author Notes

NEXUS Engine is an ongoing quantitative research project focused on systematic strategy design, portfolio construction, and robustness analysis.
