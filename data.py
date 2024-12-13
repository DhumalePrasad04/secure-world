import yfinance as yf
import requests,pandas as pd
from bs4 import BeautifulSoup
import os



def save_data(data, symbol, folder="data/raw"):
    """
    Save fetched stock data to a CSV file.

    Parameters:
        data (pd.DataFrame): Stock data to save.
        symbol (str): Stock symbol for the file name.
        folder (str): Directory to save the file.
    """
    os.makedirs(folder, exist_ok=True)
    file_path = os.path.join(folder, f"{symbol}.csv")
    data.to_csv(file_path, index=True)
    print(f"Data saved for {symbol} at {file_path}")

def fetch_from_yahoo(symbol, start_date, end_date):
    """
    Fetch historical stock data from Yahoo Finance.

    Parameters:
        symbol (str): Stock symbol (e.g., "RELIANCE.NS").
        start_date (str): Start date for the data (YYYY-MM-DD).
        end_date (str): End date for the data (YYYY-MM-DD).

    Returns:
        pd.DataFrame: DataFrame containing stock data.
    """
    try:
        data = yf.download(symbol, start=start_date, end=end_date)
        if data.empty:
            raise ValueError(f"No data found for symbol: {symbol}")
        return data
    except Exception as e:
        print(f"Error fetching data from Yahoo Finance for {symbol}: {e}")
        return None

import io

def fetch_from_alpha_vantage(symbol, api_key):
    try:
        url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&symbol={symbol}&apikey={api_key}&datatype=csv"
        response = requests.get(url)
        if response.status_code != 200:
            raise ValueError(f"Alpha Vantage API error for {symbol}: {response.status_code}")
        data = pd.read_csv(io.StringIO(response.text))
        data['timestamp'] = pd.to_datetime(data['timestamp'])
        return data
    except Exception as e:
        print(f"Error fetching data from Alpha Vantage for {symbol}: {e}")
        return None


from nsetools import Nse

def fetch_from_nse(symbol):
    try:
        nse = Nse()
        stock_data = nse.get_quote(symbol)
        if not stock_data:
            raise ValueError(f"No data found for {symbol} on NSE.")
        # Convert stock_data to DataFrame
        df = pd.DataFrame([stock_data])
        return df
    except Exception as e:
        print(f"Error fetching data from NSE for {symbol}: {e}")
        return None


def fetch_stock_data(symbols, start_date, end_date, source="yahoo", api_key=None):
    """
    Fetch stock data from multiple sources including Yahoo Finance, Alpha Vantage, NSE, and BSE.

    Parameters:
        symbols (list): List of stock symbols to fetch (e.g., ["RELIANCE.NS", "TCS.NS"]).
        start_date (str): Start date for data (YYYY-MM-DD).
        end_date (str): End date for data (YYYY-MM-DD).
        source (str): Source of data ("yahoo", "alpha_vantage", "nse", "bse").
        api_key (str): API key for Alpha Vantage (optional).

    Returns:
        dict: Dictionary of fetched DataFrames by symbol.
    """
    all_data = {}
    for symbol in symbols:
        print(f"Fetching data for {symbol} from {source}...")
        data = None

        # Fetch data based on source
        if source == "yahoo":
            data = fetch_from_yahoo(symbol, start_date, end_date)
        elif source == "alpha_vantage":
            if not api_key:
                print("API key required for Alpha Vantage.")
                continue
            data = fetch_from_alpha_vantage(symbol, api_key)
        elif source == "nse":
            data = fetch_from_nse(symbol)

        else:
            print(f"Invalid source specified: {source}")
            continue

        if data is not None:
            all_data[symbol] = data
            save_data(data, symbol, folder=f"data/raw/{source}")
        else:
            print(f"Failed to fetch data for {symbol} from {source}.")
    return all_data


if __name__ == "__main__":
    from datetime import datetime

    # Parameters
    start_date = "2021-01-01"
    end_date = datetime.today().strftime("%Y-%m-%d")
    symbols = [
        "RELIANCE.NS",  # Reliance Industries
        "TCS.NS",  # Tata Consultancy Services
        "INFY.NS",  # Infosys
        "HDFCBANK.NS",  # HDFC Bank
        "HINDUNILVR.NS",  # Hindustan Unilever
        "ITC.NS",  # ITC
        "KOTAKBANK.NS",  # Kotak Mahindra Bank
        "LT.NS",  # Larsen & Toubro
        "MARUTI.NS",  # Maruti Suzuki
        "NESTLEIND.NS",  # Nestle India
        "ONGC.NS",  # Oil and Natural Gas Corporation
        "POWERGRID.NS",  # Power Grid Corporation of India
        "SBIN.NS",  # State Bank of India
        "SUNPHARMA.NS",  # Sun Pharmaceutical Industries
        "TATASTEEL.NS",  # Tata Steel
        "TECHM.NS",  # Tech Mahindra
        "ULTRACEMCO.NS",  # UltraTech Cement
        "WIPRO.NS",  # Wipro
        "YESBANK.NS",  # Yes Bank
        "ZEEL.NS",  # Zee Entertainment Enterprises
    ]
    alpha_vantage_symbols = [
        "RELIANCE",  # Reliance Industries
        "TCS",       # Tata Consultancy Services
        "INFY",      # Infosys
        "HDFCBANK",  # HDFC Bank
        "HINDUNILVR",  # Hindustan Unilever
        "ITC",       # ITC
        "KOTAKBANK", # Kotak Mahindra Bank
        "LT",        # Larsen & Toubro
        "MARUTI",    # Maruti Suzuki
        "NESTLEIND", # Nestle India
        "ONGC",      # Oil and Natural Gas Corporation
        "POWERGRID", # Power Grid Corporation of India
        "SBIN",      # State Bank of India
        "SUNPHARMA", # Sun Pharmaceutical Industries
        "TATASTEEL", # Tata Steel
        "TECHM",     # Tech Mahindra
        "ULTRACEMCO",  # UltraTech Cement
        "WIPRO",     # Wipro
        "YESBANK",   # Yes Bank
        "ZEEL",      # Zee Entertainment Enterprises
    ]



    # BSE codes for companies
    alpha_vantage_key = "iN5ttPrL294jhwgfuM2KtUnnLVNN3Dxo"

    # Define sources and corresponding output folders
    sources = {
        "yahoo": {"symbols": symbols, "folder": "data/yahoo", "api_key": None},
        "alpha_vantage": {"symbols": alpha_vantage_symbols, "folder": "data/alpha_vantage", "api_key": alpha_vantage_key},
        "nse": {"symbols": symbols, "folder": "data/nse", "api_key": None},

    }

    # Process each source
    for source, config in sources.items():
        print(f"Fetching data from {source.capitalize()}...")
        try:
            fetched_data = fetch_stock_data(
                symbols=config["symbols"],
                start_date=start_date,
                end_date=end_date,
                source=source,
                api_key=config["api_key"],
            )
            # Save each company's data
            for symbol, data in fetched_data.items():
                save_data(data, symbol, folder=config["folder"])
        except Exception as e:
            print(f"Error while processing {source.capitalize()}: {e}")