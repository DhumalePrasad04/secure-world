import os
import pandas as pd
import numpy as np
import pandas_ta as ta
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# ==============================
# Utility Functions
# ==============================

def load_data(symbol, data_folder="D:\programming\secure world\secure-world\data\processed\yahoo"):
    """
    Load the CSV data for a specific stock symbol.

    Parameters:
        symbol (str): Stock symbol (e.g., "RELIANCE.NS").
        data_folder (str): Folder where CSV files are stored.

    Returns:
        pd.DataFrame: Loaded DataFrame.
    """
    file_path = os.path.join(data_folder, f"{symbol}.csv")
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return None
    return pd.read_csv(file_path)

def prepare_data_for_lorentzian(data):
    # Ensure 'Date' is in datetime format and set it as index
    data['Date'] = pd.to_datetime(data['Date'])
    data.set_index('Date', inplace=True)

    # Recalculate MACD
    data[['MACD', 'Signal_Line', 'MACD_Histogram']] = ta.macd(data['Close'], fast=12, slow=26, signal=9)

    # Calculate price movement: 1 for up, 0 for down
    data['Movement'] = (data['Close'].shift(-1) > data['Close']).astype(int)

    # Features: Use 'Close' and technical indicators
    features = data[['Close', 'RSI', 'MACD', 'Signal_Line', 'Upper_Band', 'Middle_Band', 'Lower_Band']].shift(1)

    # Align features and target
    aligned_data = pd.concat([features, data['Movement']], axis=1).dropna()
    features = aligned_data.iloc[:, :-1]
    target = aligned_data.iloc[:, -1]

    return features, target



def train_test_split_data(features, target):
    """
    Split the data into training and testing sets.

    Returns:
        X_train, X_test, y_train, y_test
    """
    return train_test_split(features, target, test_size=0.2, shuffle=False)

# ==============================
# Lorentzian Classification
# ==============================

def lorentzian_distance(x1, x2):
    """
    Compute the Lorentzian distance between two points.

    Parameters:
        x1, x2 (np.array): Feature vectors.

    Returns:
        float: Lorentzian distance.
    """
    return np.sum(np.log(1 + np.abs(x1 - x2)))

def train_lorentzian_classifier(X_train, y_train):
    """
    Train a custom Lorentzian Classifier using K-Nearest Neighbors approach.

    Parameters:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training target.

    Returns:
        knn (KNeighborsClassifier): Trained KNN classifier with Lorentzian distance metric.
    """
    knn = KNeighborsClassifier(n_neighbors=5, metric=lorentzian_distance)
    knn.fit(X_train, y_train)
    return knn

def evaluate_classifier(model, X_test, y_test):
    """
    Evaluate the classifier on the test set.

    Parameters:
        model: Trained classifier.
        X_test (pd.DataFrame): Test features.
        y_test (pd.Series): True labels for the test set.

    Returns:
        accuracy (float): Classification accuracy.
    """
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.2f}")
    return accuracy, y_pred

# ==============================
# Visualization
# ==============================

def visualize_predictions(y_test, y_pred, title):
    """
    Visualize the actual vs predicted movements.

    Parameters:
        y_test (pd.Series): True values for the test set.
        y_pred (np.ndarray): Predicted values for the test set.
        title (str): Title of the plot.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(y_test.values, label="Actual", color="blue")
    plt.plot(y_pred, label="Predicted", color="orange")
    plt.title(title)
    plt.legend()
    plt.show()

# ==============================
# Main Workflow
# ==============================

def process_and_train_all_symbols(symbols, data_folder="D:\programming\secure world\secure-world\data\processed\yahoo"):
    """
    Process CSV files for all stock symbols and train Lorentzian classifiers for each stock.

    Parameters:
        symbols (list): List of stock symbols (e.g., ['RELIANCE.NS', 'TCS.NS']).
        data_folder (str): Folder containing the processed data.
    """
    for symbol in symbols:
        print(f"\nProcessing symbol: {symbol}")

        # Load data
        data = load_data(symbol, data_folder)
        if data is None:
            continue

        # Prepare data
        features, target = prepare_data_for_lorentzian(data)
        X_train, X_test, y_train, y_test = train_test_split_data(features, target)

        # Train Lorentzian Classifier
        print(f"Training Lorentzian classifier for {symbol}...")
        classifier = train_lorentzian_classifier(X_train, y_train)

        # Evaluate and visualize
        print(f"Evaluating Lorentzian classifier for {symbol}...")
        accuracy, y_pred = evaluate_classifier(classifier, X_test, y_test)


    print("\nAll symbols processed and classifiers trained!")

if __name__ == "__main__":
    symbols = [
        "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "HINDUNILVR.NS",
        "ITC.NS", "KOTAKBANK.NS", "LT.NS", "MARUTI.NS", "NESTLEIND.NS",
        "ONGC.NS", "POWERGRID.NS", "SBIN.NS", "SUNPHARMA.NS", "TATASTEEL.NS",
        "TECHM.NS", "ULTRACEMCO.NS", "WIPRO.NS", "YESBANK.NS", "ZEEL.NS"
    ]
    process_and_train_all_symbols(symbols)