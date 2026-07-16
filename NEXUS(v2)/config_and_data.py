import numpy as np
import pandas as pd
import yfinance as yf

# ==============================================================================
# RUNTIME PARAMETER OBJECT ENVIRONMENT
# ==============================================================================
class SystemContext:
    """
    Encapsulates all portfolio sizing bounds, transaction expenses, 
    and date constraints. Prevents cross-module global namespace leaking
    during automated multi-regime optimization loops.
    """
    def __init__(self, 
                 start_capital=100000, 
                 risk_per_trade=0.01, 
                 max_sector_positions=2, 
                 max_total_positions=6, 
                 max_alloc_fraction=1/6,
                 target_vol=0.10,
                 start_date="2023-01-01",
                 end_date="2026-06-01"):
        
        self.START_CAPITAL = start_capital
        self.RISK_PER_TRADE = risk_per_trade
        self.MAX_SECTOR_POSITIONS = max_sector_positions
        self.MAX_TOTAL_POSITIONS = max_total_positions
        self.MAX_ALLOC_FRACTION = max_alloc_fraction
        self.TARGET_VOL = target_vol
        self.START_DATE = start_date
        self.END_DATE = end_date

        # Fixed Microstructure Constraints
        self.SLIPPAGE_ENTRY = 0.0005
        self.SLIPPAGE_EXIT = 0.0005
        self.COMMISSION = 1.0
        self.RISK_FREE_ANNUAL = 0.045
        self.DAILY_RF = self.RISK_FREE_ANNUAL / 252

# ==========================================
# STATIC ASSET REPOSITORY CONSTRAINTS
# ==========================================
tickers = [
    'MSFT', 'AMZN', 'JPM', 'UNH', 'XOM', 'WMT', 'NVDA', 'AMD', 'META', 'TSLA',
    'AAPL', 'GOOGL', 'NFLX', 'AVGO', 'CRM',  
    'HD', 'COST', 'NKE', 'MCD', 'ORLY',      
    'BAC', 'GS', 'MS', 'V', 'MA',            
    'LLY', 'MRK', 'PFE', 'JNJ',              
    'CVX', 'COP'                             
]

sectors = {
    'MSFT': 'Tech', 'NVDA': 'Tech', 'AMD': 'Tech', 'META': 'Tech',
    'AAPL': 'Tech', 'GOOGL': 'Tech', 'NFLX': 'Tech', 'AVGO': 'Tech', 'CRM': 'Tech',
    'AMZN': 'Consumer', 'TSLA': 'Consumer', 'WMT': 'Consumer',
    'HD': 'Consumer', 'COST': 'Consumer', 'NKE': 'Consumer', 'MCD': 'Consumer', 'ORLY': 'Consumer',
    'JPM': 'Finance', 'BAC': 'Finance', 'GS': 'Finance', 'MS': 'Finance',
    'V': 'Finance', 'MA': 'Finance',
    'UNH': 'Healthcare', 'LLY': 'Healthcare', 'MRK': 'Healthcare', 'PFE': 'Healthcare', 'JNJ': 'Healthcare',
    'XOM': 'Energy', 'CVX': 'Energy', 'COP': 'Energy'
}

# ==========================================
# DATA INGESTION ENGINE
# ==========================================
def load_data(ctx: SystemContext):
    print(f"Downloading historical data ({ctx.START_DATE} to {ctx.END_DATE})...")
    data = yf.download(tickers + ['^GSPC'], start=ctx.START_DATE, end=ctx.END_DATE, auto_adjust=False)
    return data

# ==========================================
# STRUCTURAL FEATURE COMPILER
# ==========================================
def build_indicators(data):
    close = data['Close']
    high = data['High']
    low = data['Low']
    vol = data['Volume']

    ind = {}

    for t in tickers:
        if t not in close.columns:
            print(f"WARNING: '{t}' returned no data -- skipping.")
            continue
        s = close[t]
        if s.isna().mean() > 0.05:
            print(f"WARNING: '{t}' has >5% missing history -- skipping.")
            continue

        df = pd.DataFrame(index=close.index)
        df['Close'] = s
        df['High'] = high[t]
        df['Low'] = low[t]
        df['Vol'] = vol[t]

        # Trend & Moving Averages
        df['SMA50'] = df['Close'].rolling(50).mean()
        df['SMA20'] = df['Close'].rolling(20).mean()
        df['EMA20'] = df['Close'].ewm(span=20).mean()
        df['SMA5'] = df['Close'].rolling(5).mean()
        df['SMA200'] = df['Close'].rolling(200).mean()

        # Volume & Channel Breakout References
        df['AvgVol30'] = df['Vol'].rolling(30).mean()
        df['Prior20High'] = df['Close'].rolling(20).max().shift(1)

        # Average True Range (Volatility Tracking)
        tr1 = df['High'] - df['Low']
        tr2 = (df['High'] - df['Close'].shift(1)).abs()
        tr3 = (df['Low'] - df['Close'].shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df['ATR14'] = tr.rolling(14).mean()
        df['ATR50'] = tr.rolling(50).mean()

        # Relative Strength Index (Exhaustion Tracking)
        delta = df['Close'].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(2).mean()
        avg_loss = loss.rolling(2).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        df['RSI2'] = 100 - (100 / (1 + rs))

        # Keltner Channels (Compression Squeeze Tracking)
        df['KC_MID'] = df['EMA20']
        df['KC_UPPER'] = df['EMA20'] + 1.5 * df['ATR14']
        df['KC_LOWER'] = df['EMA20'] - 1.5 * df['ATR14']

        ind[t] = df

    sp500 = close['^GSPC']
    sp500_sma200 = sp500.rolling(200).mean()
    sp500_sma50 = sp500.rolling(50).mean()

    return ind, sp500, sp500_sma200, sp500_sma50
