# RUN THIS IN GOOGLE COLAB

import sys
import types
import pandas as pd

# ==============================================================================
# FORCE INJECT FULL-SCALE OFFLINE DATABASE UNIVERSE INTO MEMORY (VERSION 4)
# ==============================================================================
mock_module = types.ModuleType('universe_factory')

def get_historical_spx_constituents(start_date="2007-01-01"):
    """
    Offline Local Database Engine: Reads Farrell Aultman's vertical log file 
    from disk, splitting the comma-separated text string cell data to dynamically 
    track all active historical index components across time.
    """
    try:
        # Load Farrell Aultman's vertical index tracking file
        df = pd.read_csv('sp500_full_history.csv')
    except FileNotFoundError:
        raise FileNotFoundError(
            "CRITICAL ERROR: 'sp500_full_history.csv' not found! "
            "Please upload 'S&P 500 Historical Components & Changes (Updated).csv' "
            "and rename it to 'sp500_full_history.csv' via the sidebar folder panel."
        )

    # Standardize column strings to enforce perfect case mapping
    df.columns = [str(c).strip().lower() for c in df.columns]
    
    # Locate date and ticker column locations dynamically
    date_col = 'date' if 'date' in df.columns else df.columns[0]
    tickers_col = 'tickers' if 'tickers' in df.columns else df.columns[1]
    
    df['Standard_Date'] = pd.to_datetime(df[date_col])
    df = df.sort_values(by='Standard_Date', ascending=True)
    
    # Filter for the target backtest horizon segment
    df = df[df['Standard_Date'] >= pd.to_datetime(start_date)]
    
    universe_history = {}
    
    # Iterate through the rows to split the raw text cell strings
    for _, row in df.iterrows():
        event_date = row['Standard_Date'].strftime('%Y-%m-%d')
        raw_cell_string = str(row[tickers_col])
        
        # FIXED: Split the giant cell string at every comma to isolate ticker strings
        if pd.isna(row[tickers_col]) or raw_cell_string.strip().lower() in ['nan', '']:
            day_tickers = []
        else:
            # Clean spaces and convert punctuation formatting (e.g. BRK.B -> BRK-B)
            day_tickers = [t.strip().upper().replace('.', '-') for t in raw_cell_string.split(',') if t.strip()]
            
        universe_history[event_date] = list(set(day_tickers))
        
    return universe_history

# Assign the fixed text-parsing engine to our fake module and inject it into Python's core
mock_module.get_historical_spx_constituents = get_historical_spx_constituents
sys.modules['universe_factory'] = mock_module

print("🧠 Memory hijacked successfully! Comma-separated parsing engine active.")
print("Launching full-scale point-in-time backtest suite now...\n")

# ==========================================
# TRIGGER EXPLOSIVE HARNESS SIMULATION
# ==========================================
%run main_harness.py
