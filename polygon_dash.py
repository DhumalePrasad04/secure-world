import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import ta
from datetime import datetime, timedelta

# Polygon API Configuration
API_KEY = "zdk1cDqdialHsvHM1V36LNsioCAuQ22w"  # Replace with your Polygon.io API key
BASE_URL = "https://api.polygon.io/v2/aggs/ticker"

# Function to fetch stock data using requests module
def fetch_stock_data(ticker, from_date=None, to_date=None):
    """
    Fetch historical stock data from Polygon.io using the requests module.

    Parameters:
        ticker (str): Stock ticker symbol (e.g., "AAPL").
        from_date (str): Start date in the format "YYYY-MM-DD".
        to_date (str): End date in the format "YYYY-MM-DD".

    Returns:
        pd.DataFrame: DataFrame containing stock data.
    """
    try:
        # Validate and adjust the date range if needed
        if not from_date:
            from_date = (datetime.today() - timedelta(days=30)).strftime("%Y-%m-%d")
        if not to_date:
            to_date = datetime.today().strftime("%Y-%m-%d")

        # Ensure from_date is before to_date
        if from_date > to_date:
            st.warning("Start date must be before or equal to end date. Adjusting dates.")
            from_date = to_date  # Set from_date to to_date to prevent error

        # Construct the API URL
        url = f"{BASE_URL}/{ticker}/range/1/day/{from_date}/{to_date}?apiKey={API_KEY}"
        response = requests.get(url)
        response.raise_for_status()  # Raise an error for bad status codes
        data = response.json()

        # Check if results exist
        results = data.get("results", [])
        if not results:
            st.warning(f"No data available for {ticker}.")
            return pd.DataFrame()

        # Convert results to DataFrame
        df = pd.DataFrame(results)
        df["timestamp"] = pd.to_datetime(df["t"], unit="ms")
        df = df.rename(columns={
            "o": "Open", "h": "High", "l": "Low", "c": "Close", "v": "Volume"
        })[["timestamp", "Open", "High", "Low", "Close", "Volume"]]
        return df
    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {e}")
        return pd.DataFrame()

# Function to calculate RSI
def calculate_rsi(data: pd.DataFrame, periods: int = 14) -> pd.DataFrame:
    rsi = ta.momentum.RSIIndicator(close=data['Close'], window=periods)
    data['RSI'] = rsi.rsi()
    data['RSI_Signal'] = np.where(data['RSI'] < 30, 'Buy', np.where(data['RSI'] > 70, 'Sell', 'Hold'))
    return data

# Function to calculate MACD
def calculate_macd(data: pd.DataFrame, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9) -> pd.DataFrame:
    macd = ta.trend.MACD(close=data['Close'], window_slow=slow_period, window_fast=fast_period, window_sign=signal_period)
    data['MACD'] = macd.macd()
    data['MACD_Signal'] = macd.macd_signal()
    data['MACD_Trend_Signal'] = np.where(data['MACD'] > data['MACD_Signal'], 'Buy', 'Sell')
    return data

# Function to calculate Bollinger Bands
def calculate_bollinger_bands(data: pd.DataFrame, periods: int = 20, std_dev: int = 2) -> pd.DataFrame:
    bollinger = ta.volatility.BollingerBands(close=data['Close'], window=periods, window_dev=std_dev)
    data['BB_High'] = bollinger.bollinger_hband()
    data['BB_Low'] = bollinger.bollinger_lband()
    data['BB_Middle'] = bollinger.bollinger_mavg()
    data['BB_Signal'] = np.where(data['Close'] < data['BB_Low'], 'Buy',
                                 np.where(data['Close'] > data['BB_High'], 'Sell', 'Hold'))
    return data

# Function to generate a combined signal
def generate_combined_signal(data: pd.DataFrame) -> str:
    signals = {
        'RSI_Signal': data['RSI_Signal'].iloc[-1],
        'MACD_Trend_Signal': data['MACD_Trend_Signal'].iloc[-1],
        'BB_Signal': data['BB_Signal'].iloc[-1]
    }

    buy_count = sum(1 for signal in signals.values() if signal == 'Buy')
    sell_count = sum(1 for signal in signals.values() if signal == 'Sell')

    if buy_count > sell_count:
        return 'Strong Buy'
    elif sell_count > buy_count:
        return 'Strong Sell'
    else:
        return 'Neutral'

# Function to plot candlestick chart
def plot_candlestick_chart(data, ticker):
    fig = go.Figure(data=[go.Candlestick(
        x=data["timestamp"],
        open=data["Open"],
        high=data["High"],
        low=data["Low"],
        close=data["Close"],
        increasing_line_color="green",
        decreasing_line_color="red"
    )])
    fig.update_layout(
        title=f"{ticker} Stock Price",
        xaxis_title="Date",
        yaxis_title="Price",
        template="plotly_dark"
    )
    st.plotly_chart(fig, use_container_width=True)

# Streamlit Dashboard
def main():
    st.title("Stock Dashboard with Technical Indicators")
    st.sidebar.header("Configuration")

    ticker = st.sidebar.text_input("Enter Stock Ticker (e.g., AAPL, MSFT):", value="AAPL")
    from_date = st.sidebar.date_input("Start Date", value=datetime.today() - timedelta(days=30))
    to_date = st.sidebar.date_input("End Date", value=datetime.today())

    if st.sidebar.button("Fetch Data"):
        st.write(f"Fetching data for {ticker} from {from_date} to {to_date}...")
        data = fetch_stock_data(ticker, from_date=str(from_date), to_date=str(to_date))

        if not data.empty:
            data = calculate_rsi(data)
            data = calculate_macd(data)
            data = calculate_bollinger_bands(data)

            st.subheader(f"Stock Data for {ticker}")
            st.dataframe(data)

            st.subheader(f"{ticker} Price Movement")
            plot_candlestick_chart(data, ticker)

            st.subheader("Combined Signal")
            signal = generate_combined_signal(data)
            if signal=="buy":
                st.success(signal)
            elif signal=="sell":
                st.success(signal)
            else:
                st.warning(signal)
        else:
            st.warning("No data to display.")

if __name__ == "__main__":
    main()
