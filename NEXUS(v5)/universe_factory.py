%%writefile universe_factory.py
import pandas as pd

def get_historical_spx_constituents(start_date="2007-01-01"):
    """
    Full-Scale Local Dataset Engine: Reads Farrell Aultman's horizontal grid file 
    from disk to dynamically track every stock alive on any given day.
    """
    try:
        # Load the updated master index tracking grid
        df = pd.read_csv('sp500_full_history.csv')
    except FileNotFoundError:
        raise FileNotFoundError("CRITICAL ERROR: 'sp500_full_history.csv' not found in your Colab workspace folder! Please upload 'S&P 500 Historical Components & Changes (Updated).csv' and rename it.")

    # Locate and standardize the date column name dynamically from his dataset
    date_col = next((col for col in df.columns if col.lower() == 'date'), df.columns[0])
    df['Standard_Date'] = pd.to_datetime(df[date_col])
    
    # Sort chronologically to align with your main harness loop steps
    df = df.sort_values(by='Standard_Date', ascending=True)
    df = df[df['Standard_Date'] >= pd.to_datetime(start_date)]
    
    universe_history = {}
    
    # Iterate across the 2,595 rows horizontally to harvest active tickers
    for _, row in df.iterrows():
        event_date = row['Standard_Date'].strftime('%Y-%m-%d')
        
        # Pull every value from the row, ignoring metadata columns and empty fields
        raw_tickers = []
        for col, val in row.items():
            if col != 'Standard_Date' and col != date_col and pd.notna(val):
                ticker_str = str(val).strip().upper().replace('.', '-')
                # Ensure it's a clean, valid stock symbol ticker format string
                if ticker_str and len(ticker_str) <= 5 and ticker_str.isalpha():
                    raw_tickers.append(ticker_str)
                    
        # Lock in today's unique roster array list
        universe_history[event_date] = list(set(raw_tickers))
        
    return universe_history
