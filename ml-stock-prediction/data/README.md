# Overview of the Data Directory

This directory contains the datasets used for the Machine Learning-Based Stock Market Prediction project. The data is crucial for training, validating, and testing the machine learning models implemented in this project.

## Datasets

1. **Historical Stock Prices**
   - **Description**: This dataset contains historical stock prices for various companies. It includes daily open, high, low, close prices, and volume traded.
   - **Source**: Data is sourced from Yahoo Finance using the `yfinance` library.
   - **Format**: CSV (Comma-Separated Values)
   - **File Name**: `historical_stock_prices.csv`

2. **Technical Indicators**
   - **Description**: This dataset includes calculated technical indicators such as Moving Averages, RSI, MACD, etc., which are used as features for the machine learning models.
   - **Source**: Generated from the historical stock prices using the `ta` library.
   - **Format**: CSV
   - **File Name**: `technical_indicators.csv`

3. **Market Sentiment Data**
   - **Description**: This dataset contains sentiment analysis scores derived from news articles and social media related to the stock market.
   - **Source**: Collected using web scraping techniques or APIs.
   - **Format**: CSV
   - **File Name**: `market_sentiment.csv`

## Usage

The datasets should be placed in this directory to ensure that the scripts in the `src` folder can access them for data preprocessing, feature engineering, model training, and evaluation. Make sure to keep the data updated for accurate predictions.