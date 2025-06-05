# filepath: ml-stock-prediction/ml-stock-prediction/src/feature_engineering.py

import pandas as pd
import numpy as np
import ta  # Technical Analysis Library

def create_lagged_features(data, lag=1):
    """
    Create lagged features for the dataset.
    
    Parameters:
    data (pd.DataFrame): The input dataframe containing stock prices.
    lag (int): The number of lagged periods to create.
    
    Returns:
    pd.DataFrame: DataFrame with lagged features added.
    """
    for i in range(1, lag + 1):
        data[f'lag_{i}'] = data['close'].shift(i)
    return data

def calculate_technical_indicators(data):
    """
    Calculate technical indicators and add them to the dataset.
    
    Parameters:
    data (pd.DataFrame): The input dataframe containing stock prices.
    
    Returns:
    pd.DataFrame: DataFrame with technical indicators added.
    """
    data['SMA_20'] = data['close'].rolling(window=20).mean()
    data['SMA_50'] = data['close'].rolling(window=50).mean()
    data['EMA_20'] = data['close'].ewm(span=20, adjust=False).mean()
    data['RSI'] = ta.momentum.RSIIndicator(data['close']).rsi()
    data['MACD'] = ta.trend.MACD(data['close']).macd()
    return data

def perform_feature_selection(data, target):
    """
    Perform feature selection based on correlation with the target variable.
    
    Parameters:
    data (pd.DataFrame): The input dataframe containing features.
    target (str): The target variable name.
    
    Returns:
    pd.DataFrame: DataFrame with selected features.
    """
    correlation_matrix = data.corr()
    relevant_features = correlation_matrix[target].abs().sort_values(ascending=False)
    selected_features = relevant_features[relevant_features > 0.1].index.tolist()
    return data[selected_features]