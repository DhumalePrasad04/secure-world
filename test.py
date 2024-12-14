import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import ta,threading
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from flask import Flask, jsonify


# Polygon API Configuration
API_KEY = "zdk1cDqdialHsvHM1V36LNsioCAuQ22w"  # Replace with your Polygon.io API key
BASE_URL = "https://api.polygon.io/v2/aggs/ticker"

companies = {
    "Apple Inc.": "AAPL",
    "Microsoft Corporation": "MSFT",
    "Amazon.com Inc.": "AMZN",
    "Alphabet Inc. (Google)": "GOOGL",
    "Meta Platforms Inc. (Facebook)": "META",
    "Tesla Inc.": "TSLA",
    "Berkshire Hathaway Inc.": "BRK.B",
    "Johnson & Johnson": "JNJ",
    "Visa Inc.": "V",
    "Procter & Gamble Co.": "PG",
    "NVIDIA Corporation": "NVDA",
    "JPMorgan Chase & Co.": "JPM",
    "UnitedHealth Group Incorporated": "UNH",
    "Walt Disney Company": "DIS",
    "Home Depot Inc.": "HD",
    "Pfizer Inc.": "PFE",
    "Exxon Mobil Corporation": "XOM",
    "Chevron Corporation": "CVX",
    "Coca-Cola Company": "KO",
    "PepsiCo Inc.": "PEP",
    "McDonald's Corporation": "MCD",
    "Nike Inc.": "NKE",
    "Intel Corporation": "INTC",
    "AbbVie Inc.": "ABBV",
    "Verizon Communications Inc.": "VZ",
    "AT&T Inc.": "T",
    "IBM Corporation": "IBM",
    "Caterpillar Inc.": "CAT",
    "Lockheed Martin Corporation": "LMT",
    "Bristol-Myers Squibb Company": "BMY",
    "Goldman Sachs Group Inc.": "GS",
    "Wells Fargo & Co.": "WFC",
    "Morgan Stanley": "MS",
    "Salesforce Inc.": "CRM",
    "The Boeing Company": "BA",
    "Oracle Corporation": "ORCL",
    "American Express Company": "AXP",
    "Walmart Inc.": "WMT",
    "3M Company": "MMM",
    "Starbucks Corporation": "SBUX",
    "General Electric Company": "GE",
    "Target Corporation": "TGT",
    "The Kraft Heinz Company": "KHC",
    "Raytheon Technologies Corporation": "RTX",
    "Citigroup Inc.": "C",
    "Honeywell International Inc.": "HON",
    "General Motors Company": "GM",
    "Bristol Myers Squibb": "BMY"
}

# Function to fetch stock data using requests module
def fetch_stock_data(ticker, from_date=None, to_date=None):
    try:
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

def calculate_indicators(data: pd.DataFrame) -> pd.DataFrame:
    if data.empty:
        return data

    # RSI
    rsi = ta.momentum.RSIIndicator(close=data['Close'], window=14)
    data['RSI'] = rsi.rsi()
    data['RSI_Signal'] = np.where(data['RSI'] < 30, 'Buy', np.where(data['RSI'] > 70, 'Sell', 'Hold'))

    # MACD
    macd = ta.trend.MACD(close=data['Close'], window_slow=26, window_fast=12, window_sign=9)
    data['MACD'] = macd.macd()
    data['MACD_Signal'] = macd.macd_signal()
    data['MACD_Trend_Signal'] = np.where(data['MACD'] > data['MACD_Signal'], 'Buy', 'Sell')

    # Bollinger Bands
    bb = ta.volatility.BollingerBands(close=data['Close'], window=20, window_dev=2)
    data['BB_High'] = bb.bollinger_hband()
    data['BB_Low'] = bb.bollinger_lband()
    data['BB_Signal'] = np.where(data['Close'] < data['BB_Low'], 'Buy',
                                 np.where(data['Close'] > data['BB_High'], 'Sell', 'Hold'))

    # Combined Signal
    data['Combined_Signal'] = generate_combined_signal(data)

    return data
app=Flask(__name__)
@app.route('/api/stock_data', methods=['GET'])
def get_stock_data():
    ticker = "AAPL"  # Example: Fetch data for Apple (can be parameterized)
    data = fetch_stock_data(ticker)
    if not data.empty:
        data = calculate_indicators(data)
        last_row = data.iloc[-1].to_dict()
        return jsonify(last_row)
    else:
        return jsonify({"error": "No stock data available"}), 404
def run_flask():
    app.run(host="127.0.0.1", port=5000, debug=False)

# Streamlit Dashboard
def main():
    st.title("Stock Dashboard with Technical Indicators")
    st.sidebar.header("Configuration")

    ticker = st.sidebar.selectbox("Select Stock", list(companies.keys()), index=0)
    from_date = st.sidebar.date_input("Start Date", value=datetime.today() - timedelta(days=30))
    to_date = st.sidebar.date_input("End Date", value=datetime.today())

    if st.sidebar.button("Fetch Data"):
        st.write(f"Fetching data for {ticker} from {from_date} to {to_date}...")
        ticker_symbol = companies[ticker]  # Get the ticker symbol from the dictionary
        data = fetch_stock_data(ticker_symbol, from_date=str(from_date), to_date=str(to_date))

        if not data.empty:
            data = calculate_rsi(data)
            data = calculate_macd(data)
            data = calculate_bollinger_bands(data)

            # Stock Data Table
            st.subheader(f"Stock Data for {ticker}")
            st.dataframe(data)

            # Price Movement (Candlestick Chart)
            st.subheader(f"{ticker} Price Movement")
            plot_candlestick_chart(data, ticker)

            # RSI (Relative Strength Index)
            st.subheader(f"{ticker} RSI")
            st.line_chart(data.set_index("timestamp")["RSI"])

            # Trading Volume Bar Chart
            st.subheader(f"{ticker} Trading Volume")
            st.bar_chart(data.set_index("timestamp")["Volume"])

            # In the 'main' function, replace the histogram plotting section with this:
            st.subheader(f"{ticker} Closing Price Distribution")
            fig, ax = plt.subplots(figsize=(8, 6))  # Adjust figure size as needed
            sns.histplot(data['Close'], bins=50, kde=True, color='blue', ax=ax)
            ax.set_title(f"{ticker} Closing Price Distribution")
            st.pyplot(fig)

            # Display combined signal
            combined_signal = generate_combined_signal(data)
            st.subheader(f"Combined Trading Signal")
            st.write(f"The combined trading signal for {ticker} is: {combined_signal}")
        else:
            st.warning("No data available for the selected stock.")

if __name__ == "__main__":
    threading.Thread(target=run_flask, daemon=True).start()
    main()
