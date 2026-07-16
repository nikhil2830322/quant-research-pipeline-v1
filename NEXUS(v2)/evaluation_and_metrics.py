import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from config_and_data import SystemContext

# ==========================================
# BENCHMARK - SPX BUY & HOLD
# ==========================================
def run_benchmark(sp500, dates, ctx: SystemContext):
    """
    Buy-and-hold the index over the identical trading horizon.
    Guarantees structural index length alignment across strategy curves.
    """
    valid_dates = list(dates)
    base_price = sp500.loc[valid_dates[0]]
    
    equity_curve = []
    daily_returns = []
    prev_equity = ctx.START_CAPITAL

    for i in range(1, len(valid_dates)):
        date = valid_dates[i]
        # Track continuous proportional change from the absolute initial entry frame
        equity = ctx.START_CAPITAL * (sp500.loc[date] / base_price)
        
        # Calculate daily percentage shifts safely
        ret = (equity - prev_equity) / prev_equity if prev_equity != 0 else 0.0
        daily_returns.append(ret)
        equity_curve.append(equity)
        prev_equity = equity

    return {
        'name': 'SPX Buy & Hold', 
        'equity_curve': equity_curve, 
        'trade_log': [],
        'daily_returns': daily_returns, 
        'dates': valid_dates[1:]
    }

# ==============================================================================
# PERFORMANCE METRICS ENGINE
# ==============================================================================
def compute_metrics(model, ctx: SystemContext):
    eq = pd.Series(model['equity_curve'])
    daily = np.array(model['daily_returns'])
    trades = model.get('trade_log', [])

    # Max Drawdown Tracking
    running_peak = eq.cummax()
    drawdown = (eq - running_peak) / running_peak if not running_peak.empty else eq
    max_dd = drawdown.min() * 100 if not drawdown.empty else 0.0

    # Return Metrics
    total_return = (eq.iloc[-1] - ctx.START_CAPITAL) / ctx.START_CAPITAL * 100 if not eq.empty else 0.0

    # Risk-Adjusted Standard Deviations
    daily_std = np.std(daily)
    sharpe = ((np.mean(daily) - ctx.DAILY_RF) / daily_std) * np.sqrt(252) if daily_std > 0 else 0.0

    # Downside Sortino Dev Calculations
    downside_returns = np.where(daily < 0, daily, 0.0)
    downside_dev = np.std(downside_returns)
    sortino = ((np.mean(daily) - ctx.DAILY_RF) / downside_dev) * np.sqrt(252) if downside_dev > 0 else 0.0

    # Profit Factor Optimization Pass
    for tr in trades:
        if tr['entry_price'] > 0:
            tr['pct_return'] = (tr['exit_price'] - tr['entry_price']) / tr['entry_price']
        else:
            tr['pct_return'] = 0.0

    gross_pct_profit = sum(tr['pct_return'] for tr in trades if tr['pct_return'] > 0)
    gross_pct_loss = -sum(tr['pct_return'] for tr in trades if tr['pct_return'] < 0)
    
    # Secure profit factor boundary condition
    if gross_pct_loss > 0:
        profit_factor = gross_pct_profit / gross_pct_loss
    else:
        profit_factor = np.inf if gross_pct_profit > 0 else 1.0

    # Win Rate Tracking
    wins = sum(1 for tr in trades if tr['pct_return'] > 0)
    total_trades = len(trades)
    win_rate = (wins / total_trades * 100) if total_trades > 0 else 0.0

    return {
        'name': model['name'], 'total_return': total_return, 'max_dd': max_dd,
        'sharpe': sharpe, 'sortino': sortino, 'profit_factor': profit_factor,
        'win_rate': win_rate, 'total_trades': total_trades,
        'equity_curve': eq, 'drawdown': drawdown, 'trade_log': trades
    }

def compare_models(results, ctx: SystemContext, SHOW_CHARTS=False):
    metrics = [compute_metrics(r, ctx) for r in results]

    if SHOW_CHARTS:
        plt.figure(figsize=(12, 6))
        for m in metrics:
            plt.plot(m['equity_curve'], label=m['name'])
        plt.title("NEXUS Engine Framework - Equity Curves")
        plt.legend()
        plt.grid(True)
        plt.show()

    table = pd.DataFrame({
        'Model': [m['name'] for m in metrics],
        'Total Return %': [m['total_return'] for m in metrics],
        'Max Drawdown %': [m['max_dd'] for m in metrics],
        'Sharpe': [m['sharpe'] for m in metrics],
        'Sortino': [m['sortino'] for m in metrics],
        'Profit Factor': [m['profit_factor'] for m in metrics],
        'Win Rate %': [m['win_rate'] for m in metrics],
        'Trades': [m['total_trades'] for m in metrics]
    })

    print("\n==============================================================")
    print("MODEL PERFORMANCE COMPARISON (including SPX benchmark)")
    print("==============================================================")
    print(table.to_string(index=False, float_format=lambda x: f"{x:.2f}"))

    print("\n==============================================================")
    print("STRATEGY RANKING (by Total Return)")
    print("==============================================================")
    ranked = table.sort_values(by='Total Return %', ascending=False)
    print(ranked[['Model', 'Total Return %']].to_string(index=False, float_format=lambda x: f"{x:.2f}"))

    bench_row = table[table['Model'] == 'SPX Buy & Hold']
    if not bench_row.empty:
        bench_return = bench_row['Total Return %'].values[0]
        bench_sharpe = bench_row['Sharpe'].values[0]
        print("\n==============================================================")
        print("ALPHA OVER SPX BUY & HOLD")
        print("==============================================================")
        for m in metrics:
            if m['name'] == 'SPX Buy & Hold':
                continue
            excess_return = m['total_return'] - bench_return
            excess_sharpe = m['sharpe'] - bench_sharpe
            print(f"{m['name']:<20}: {excess_return:+7.2f}% return vs SPX | {excess_sharpe:+.2f} Sharpe vs SPX")

    for m in metrics:
        if not m['trade_log']:
            continue
        df = pd.DataFrame(m['trade_log'])
        csv_name = f"{m['name'].lower().replace(' ', '_').replace('(', '').replace(')', '')}_trades.csv"
        df.to_csv(csv_name, index=False)
        print(f"Exported {m['name']} trades -> {csv_name}")

    return table
