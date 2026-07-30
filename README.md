Nexus Engine v5
Point-in-Time Quantitative Research Framework
Nexus Engine v5 is a research framework for testing systematic equity strategies under realistic historical constraints. The project was built around a simple question:

Can a small set of rule-based trading hypotheses generate repeatable risk-adjusted returns when tested without hindsight bias?

The engine combines three independent strategy models — Yang, Yin, and Zhan — with a regime-aware portfolio allocator, transaction cost modeling, point-in-time universe handling, and Monte Carlo stress testing.

The goal is not to predict markets. The goal is to build a controlled environment where trading hypotheses can be tested, rejected, refined, and compared.

Research Philosophy
Most backtests fail because they accidentally answer the wrong question. Common issues include survivorship bias, unrealistic execution assumptions, excessive parameter tuning, and ignoring portfolio-level behavior.

Nexus Engine focuses on controlling those failure points:

Historical universe construction attempts to avoid selecting only today's winners.
Indicators are calculated using only information available before each decision point.
Entries and exits are executed using next-day market mechanics rather than same-day assumptions.
Transaction friction, commissions, cash drag, and gap risk are modeled explicitly.
Strategies are evaluated against both absolute performance and benchmark-relative outcomes.
The system is designed as a hypothesis testing environment rather than a prediction model.

Architecture Overview
The engine is separated into several components:

Strategy Layer
Three independent trading models represent different market behaviors:

Yang — Trend Following
Yang attempts to capture persistent upward momentum.

The model focuses on:

Breakouts above previous highs
Relative volume confirmation
Moving-average trend alignment
Volatility-adjusted stops
The hypothesis:

Strong companies showing expanding momentum and participation may continue outperforming after confirmation.

Yin — Mean Reversion
Yin targets temporary price dislocations inside longer-term uptrends.

The model focuses on:

Extreme short-term RSI weakness
Long-term trend confirmation
Volatility exhaustion filters
Short holding periods
The hypothesis:

Healthy assets experiencing temporary oversold conditions may revert toward their short-term equilibrium.

Zhan — Volatility Expansion
Zhan focuses on volatility contraction followed by directional expansion.

The model uses:

Keltner Channel breakouts
EMA structure
Volume confirmation
ATR-based risk management
The hypothesis:

Periods of compression can create conditions where directional expansion becomes more probable.

Ensemble Engine
The ensemble layer combines the three models instead of assuming one strategy works in every environment.

The allocator classifies the market into three regimes:

Regime	Allocation Preference	Reasoning
Bull	Yang weighted highest, Zhan secondary, Yin disabled	Strong trends favor momentum continuation
Bear	Yin and defensive positioning favored	Avoids chasing broad market weakness
Chop	Zhan prioritized, Yin secondary	Rangebound markets require selective setups
These allocations are intentionally asymmetric. The system does not attempt to give every strategy equal influence because different market environments reward different behaviors.

The design decision was to allow strategies to become inactive when their edge is theoretically weakest rather than forcing constant participation.

Risk Management
Risk controls operate at the portfolio level rather than only at the trade level.

The framework includes:

ATR-based position sizing
Maximum portfolio positions
Sector concentration limits
Portfolio drawdown scaling
Volatility-based exposure adjustment
Cash drag modeling
Entry and exit friction
Overnight gap handling
Position size is determined by the smaller constraint between:

Maximum loss allowed per trade
Maximum portfolio allocation
This prevents large positions from forming solely because volatility is temporarily low.

Point-in-Time Universe Construction
A major source of historical backtest error is survivorship bias.

For example, testing only today's largest companies ignores companies that previously existed in the index but later disappeared.

Nexus Engine attempts to address this through historical constituent tracking.

The universe system supports:

Historical membership changes
IPO availability checks
Dynamic asset availability
Removal of unavailable securities from earlier periods
Current testing uses a controlled historical basket rather than claiming to represent the complete historical S&P 500 universe.

Execution Model
The engine uses next-day execution assumptions:

Signal generated using previous close data.
Entry occurs using the following day's opening price.
Exit logic evaluates previous information and executes using available market prices.
Execution friction includes:

Entry slippage
Exit slippage
Commission costs
Gap-down stop behavior
This prevents same-bar execution advantages that are unavailable in real trading.

Monte Carlo Stress Testing
Backtest returns are not evaluated only from one historical sequence.

The engine applies stationary block bootstrap Monte Carlo simulations.

Parameters:

5,000 simulations
Five-day return blocks
Synchronized benchmark comparisons
The simulations measure:

Return distribution
Drawdown distribution
Sharpe distribution
Probability of positive outcomes
Probability of outperforming the benchmark path
Monte Carlo results are treated as robustness analysis, not proof of future performance.

Example Validation Results
Testing period:

2007-01-01 to 2010-01-01

Market environment:

Included the 2008 financial crisis
Compared against S&P 500 buy-and-hold performance
Model	Return	Max Drawdown	Sharpe	Trades
Yang (Manual Limits)	25.23%	-6.48%	0.63	69
Zhan (Manual Limits)	13.95%	-6.96%	0.21	74
Yin (Manual Limits)	23.87%	-9.70%	0.65	260
Ensemble	18.17%	-7.97%	0.39	95
S&P 500 Buy & Hold	-27.59%	-56.34%	-0.40	—
The individual strategies and ensemble should not be interpreted as guarantees of future returns. These results represent a single historical sample used to evaluate whether the underlying hypotheses behaved as expected.

Current Limitations
This project is still a research system.

Known limitations:

The current point-in-time universe is a controlled dataset, not a complete historical reconstruction of every S&P 500 constituent.
Parameter selection has not been fully separated into training and validation periods.
Additional market environments are required before drawing conclusions about robustness.
Real-world execution would introduce additional constraints including liquidity, spreads, and market impact.
Future Development
Potential extensions:

Full historical S&P 500 constituent database integration
Walk-forward optimization
Out-of-sample validation periods
More realistic liquidity constraints
Portfolio correlation controls
Additional asset classes
Live paper
