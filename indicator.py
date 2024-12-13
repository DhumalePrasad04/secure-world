import pandas as pd
import pandas_ta as ta
import os

def clean_csv_data(data):
    """
    Clean and preprocess the stock data based on the following instructions:
    - Remove the first row's 'Price' value and replace it with 'Date'.
    - Remove the second row which contains the 'Ticker' values.
    - Remove unnecessary columns and ensure correct column names.

    Parameters:
        data (pd.DataFrame): Raw stock data from the CSV.

    Returns:
        pd.DataFrame: Cleaned and preprocessed stock data.
    """

    # Drop the first unnecessary column ('Unnamed: 0') if it exists
    data = data.drop(columns=["Unnamed: 0"], errors='ignore')

    # Check the number of columns in the raw data
    num_columns = len(data.columns)
    expected_columns = 6  # We expect 6 columns (Date, Open, High, Low, Close, Volume)

    # If there are more columns than expected, drop the extra ones
    if num_columns > expected_columns:
        data = data.iloc[:, :expected_columns]  # Keep only the first 6 columns

    # Set correct column names
    data.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']

    # Remove the first row (which contains invalid data like 'Ticker')
    data = data[1:]

    # Strip any extra spaces or commas from column names
    data.columns = data.columns.str.strip()  # Remove leading/trailing spaces
    data.columns = data.columns.str.replace(',', '')  # Remove any commas that could cause misalignment

    # Ensure 'Date' is in datetime format and handle missing or malformed values
    data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
    data.dropna(subset=['Date'], inplace=True)  # Drop rows where 'Date' could not be parsed

    # Convert other columns to numeric
    data['Open'] = pd.to_numeric(data['Open'], errors='coerce')
    data['High'] = pd.to_numeric(data['High'], errors='coerce')
    data['Low'] = pd.to_numeric(data['Low'], errors='coerce')
    data['Close'] = pd.to_numeric(data['Close'], errors='coerce')
    data['Volume'] = pd.to_numeric(data['Volume'], errors='coerce')

    return data

def calculate_indicators(data, config=None):
    """
    Calculate multiple indicators: RSI, MACD, Bollinger Bands.

    Parameters:
        data (pd.DataFrame): Preprocessed stock data.
        config (dict): Configuration for indicators (e.g., periods, deviations).

    Returns:
        pd.DataFrame: Data with calculated indicators.
    """
    if config is None:
        config = {
            "rsi": {"period": 14},
            "macd": {"fast_period": 12, "slow_period": 26, "signal_period": 9},
            "bollinger": {"period": 20, "deviation": 2},
        }

    # Calculate RSI
    if "rsi" in config:
        if 'Close' in data.columns:
            data['RSI'] = ta.rsi(data['Close'], length=config['rsi']['period'])
        else:
            print("Warning: Close column missing for RSI calculation.")

    # Calculate MACD
    if "macd" in config:
        if 'Close' in data.columns:
            macd, signal, _ = ta.macd(
                data['Close'],
                fast=config['macd']['fast_period'],
                slow=config['macd']['slow_period'],
                signal=config['macd']['signal_period']
            )
            data['MACD'] = macd
            data['Signal_Line'] = signal
        else:
            print("Warning: Close column missing for MACD calculation.")

    # Calculate Bollinger Bands
    if "bollinger" in config:
        if 'Close' in data.columns:
            bbands = ta.bbands(
                data['Close'],
                length=config['bollinger']['period'],
                nbdevup=config['bollinger']['deviation'],
                nbdevdn=config['bollinger']['deviation']
            )
            data['Upper_Band'] = bbands['BBU_20_2.0']
            data['Middle_Band'] = bbands['BBM_20_2.0']
            data['Lower_Band'] = bbands['BBL_20_2.0']
        else:
            print("Warning: Close column missing for Bollinger Bands calculation.")

    return data

def recalculate_indicators(data):
    """
    Recalculate MACD, Signal Line, and other missing indicators if not present.

    Parameters:
        data (pd.DataFrame): Raw or partially processed stock data.

    Returns:
        pd.DataFrame: Data with recalculated indicators.
    """
    if 'MACD' not in data.columns or data['MACD'].isnull().all():
        macd = ta.macd(data['Close'], fast=12, slow=26, signal=9)
        data['MACD'] = macd['MACD_12_26_9']
        data['Signal_Line'] = macd['MACDs_12_26_9']
        data['MACD_Histogram'] = macd['MACDh_12_26_9']

    if 'RSI' not in data.columns or data['RSI'].isnull().all():
        data['RSI'] = ta.rsi(data['Close'], length=14)

    if 'Upper_Band' not in data.columns or data['Upper_Band'].isnull().all():
        bbands = ta.bbands(data['Close'], length=20, nbdevup=2, nbdevdn=2)
        data['Upper_Band'] = bbands['BBU_20_2.0']
        data['Middle_Band'] = bbands['BBM_20_2.0']
        data['Lower_Band'] = bbands['BBL_20_2.0']

    return data

def process_downloaded_data(symbols, folder="data/raw/yahoo", output_folder="data/processed/yahoo", config=None):
    """
    Process downloaded stock data: Clean, calculate indicators, and save results.

    Parameters:
        symbols (list): List of stock symbols to process.
        folder (str): Folder where raw CSV files are stored.
        output_folder (str): Folder to save processed data.
        config (dict): Configuration for indicators.

    Returns:
        dict: Dictionary of processed DataFrames by symbol.
    """
    all_data = {}
    os.makedirs(output_folder, exist_ok=True)

    for symbol in symbols:
        file_path = os.path.join(folder, f"{symbol}.csv")

        if not os.path.exists(file_path):
            print(f"File not found for {symbol}. Skipping...")
            continue

        print(f"Processing data for {symbol} from {file_path}...")

        try:
            # Load and clean the data
            raw_data = pd.read_csv(file_path)
            data = clean_csv_data(raw_data)

            # Calculate indicators
            data = calculate_indicators(data, config)
            data=recalculate_indicators(data)
            # Save processed data
            output_path = os.path.join(output_folder, f"{symbol}.csv")
            data.to_csv(output_path, index=True)
            print(f"Processed data saved for {symbol} at {output_path}")

            all_data[symbol] = data

        except Exception as e:
            print(f"Error processing data for {symbol}: {e}")

    return all_data


if __name__ == "__main__":
    # Symbols to process
    symbols = [
        "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "HINDUNILVR.NS",
        "ITC.NS", "KOTAKBANK.NS", "LT.NS", "MARUTI.NS", "NESTLEIND.NS",
        "ONGC.NS", "POWERGRID.NS", "SBIN.NS", "SUNPHARMA.NS", "TATASTEEL.NS",
        "TECHM.NS", "ULTRACEMCO.NS", "WIPRO.NS", "YESBANK.NS", "ZEEL.NS"
    ]

    # Indicator configuration
    indicator_config = {
        "rsi": {"period": 14},
        "macd": {"fast_period": 12, "slow_period": 26, "signal_period": 9},
        "bollinger": {"period": 20, "deviation": 2},
    }

    # Process the data
    processed_data = process_downloaded_data(
        symbols=symbols,
        folder="data/raw/yahoo",
        output_folder="data/processed/yahoo",
        config=indicator_config
    )

    print("Processing complete. Check processed data.")
