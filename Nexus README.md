# NEXUS Engine Framework (v4)

An institutional-style quantitative trading framework built in Python for realistic portfolio simulation, strategy research, and statistical validation.

## Overview

NEXUS Engine v4 is a modular backtesting framework that focuses on realistic market simulation rather than optimistic results. The engine reconstructs the historical S&P 500 on any trading day, allowing strategies to trade only the companies that actually existed in the index at that time. This removes survivorship bias and produces far more reliable historical testing.

The framework combines multiple independent trading strategies with portfolio-level risk management, dynamic capital allocation, and Monte Carlo stress testing to evaluate long-term robustness across different market environments.

## Features

- Point-in-Time (PIT) S&P 500 reconstruction
- Eliminates survivorship bias
- Multi-strategy portfolio architecture
- Trend-following, mean-reversion, and volatility breakout models
- Dynamic position sizing and capital allocation
- Market regime filtering (Bull / Bear / Chop)
- Commission and market friction modeling
- Portfolio-level risk management
- 5,000-run Stationary Block Bootstrap Monte Carlo analysis
- Optimized matrix-based data engine for fast simulations

## Architecture

```
Universe Builder
      ↓
Data & Configuration
      ↓
Risk Management
      ↓
Trading Models
      ↓
Portfolio Engine
      ↓
Performance Analytics
      ↓
Monte Carlo Validation
```

## Included Strategies

- **Yang** – Long-term trend following using Donchian breakouts
- **Yin** – Short-term mean reversion using RSI signals
- **Zhan** – Volatility breakout strategy using Bollinger Band and Keltner Channel squeezes

## Performance

The framework has been tested across multiple market regimes, including the 2007–2010 Financial Crisis and the 2023–2026 technology-led bull market.

Highlights include:

- Reduced drawdowns compared to buy-and-hold during strong bull markets
- Positive returns during the 2008 financial crisis
- Portfolio validation using 5,000 Monte Carlo simulations
- Statistical evaluation using Sharpe Ratio, Sortino Ratio, Drawdown, Profit Factor, and Win Rate

## Design Goals

NEXUS was built around a few core principles:

- Produce realistic backtests
- Remove common sources of historical bias
- Support multiple independent alpha sources
- Manage risk at the portfolio level
- Validate results statistically rather than relying on a single equity curve

## Tech Stack

- Python
- Pandas
- NumPy
- Yahoo Finance
- Object-Oriented Design
- Vectorized Data Processing
- Monte Carlo Simulation
