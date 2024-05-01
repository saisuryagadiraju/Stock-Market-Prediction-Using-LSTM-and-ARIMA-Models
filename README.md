# AIT-Final-Project-Sai-Surya-Gadiraju-Team-3

### This README file provides an explanation of the code used to forecast stock prices for specific tickers using the ARIMA (AutoRegressive Integrated Moving Average) model. The project imports historical stock price data, scales the data, fits ARIMA models to predict future prices, and saves the forecast results to a CSV file.

## Table of Contents
1. [Introduction](#introduction)
2. [Dependencies](#dependencies)
3. [Dataset](#dataset)
4. [Data Processing](#data-processing)
5. [ARIMA Model](#arima-model)
6. [Forecasting](#forecasting)
7. [Output](#output)
8. [Usage](#usage)

## Introduction
This project aims to forecast stock prices for specific companies over a given period using the ARIMA model. The predictions are made for a week ahead, based on existing stock price data.

## Dependencies
The following Python libraries are used in this project:

#
```
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.arima.model import ARIMA
from datetime import datetime, timedelta
```
#
pandas: For data manipulation and analysis.

numpy: For numerical operations.

sklearn.preprocessing.MinMaxScaler: For data scaling.

statsmodels.tsa.arima.model.ARIMA: For ARIMA modeling.

datetime: For working with dates and times.

These packages are installed in your Python environment before running the code.
#

Dataset
The code loads historical stock price data from a CSV file named World_Stock_Prices.csv. It contains various stock tickers along with corresponding price data (Open, Close, High, Low).

Load Data: The pd.read_csv function is used to load the dataset into a DataFrame named stock_data.
Convert Date to Datetime: The pd.to_datetime function converts the 'Date' column to a datetime format for easier manipulation.
Filter Specific Tickers: The code selects a subset of stock tickers from interested_tickers.


## Loading the dataset into the stock_data data frame
stock_data = pd.read_csv("World_Stock_Prices.csv")

stock_data['Date'] = pd.to_datetime(stock_data['Date'])

# Focus on specific tickers
interested_tickers = ['BAMXF', 'AAPL', 'AMZN', 'NFLX', 'ZI', 'JPM', 'AXP', 'FDX', 'MCD']
stock_data = stock_data[stock_data['Ticker'].isin(interested_tickers)]

# Filter data for date range up to the current date
stock_data = stock_data[stock_data['Date'] <= datetime.now()]

# Function to scale data
def scale_data(data):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data.reshape(-1, 1))
    return scaled_data, scaler

# Function to fit ARIMA and make predictions
def fit_predict_arima(scaled_series, order, forecast_days):
    model = ARIMA(scaled_series, order=order)
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=forecast_days)
    return forecast

results = []

for ticker in interested_tickers:
    print(f"Processing {ticker}...")
    data = stock_data[stock_data['Ticker'] == ticker]
    data = data.sort_values('Date')
    data.set_index('Date', inplace=True)
    
    forecasts = {}
    for column in ['Open', 'Close', 'High', 'Low']:
        prices = data[column].values
        scaled_prices, scaler = scale_data(prices)
        
        # Fit ARIMA model and forecast
        try:
            forecast_scaled = fit_predict_arima(scaled_prices, order=(1, 1, 1), forecast_days=7)
            forecast = scaler.inverse_transform(forecast_scaled.reshape(-1, 1)).flatten()
            forecasts[column] = forecast
        except Exception as e:
            print(f"Failed to fit ARIMA for {ticker} {column}: {e}")
            continue
    
    # Calculate the start date for the forecast (day after the last available date)
    start_forecast_date = data.index.max() + timedelta(days=1)
    
    # Compile forecast data
    for i in range(7):
        forecast_date = start_forecast_date + timedelta(days=i)
        results.append({
            'Date': forecast_date,
            'Ticker': ticker,
            'Open': forecasts.get('Open', [None])[i],
            'Close': forecasts.get('Close', [None])[i],
            'High': forecasts.get('High', [None])[i],
            'Low': forecasts.get('Low', [None])[i]
        })

# Convert results to DataFrame and sort by Date
forecast_df = pd.DataFrame(results)
forecast_df = forecast_df.sort_values(by=['Date', 'Ticker'])
forecast_df.to_csv('specific_tickers_forecasts.csv', index=False)
print("Forecasts for specified tickers saved to 'specific_tickers_forecasts.csv'.")
