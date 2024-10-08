# AIT-Final-Project-Sai-Surya-Gadiraju-Team-3

### This README file provides an explanation of the code used to forecast stock prices for specific tickers using the ARIMA (AutoRegressive Integrated Moving Average) model. The project imports historical stock price data, scales the data, fits ARIMA models to predict future prices, and saves the forecast results to a CSV file.


# Introduction
This project aims to forecast stock prices for specific companies over a given period using the ARIMA model. The predictions are made for a week ahead, based on existing stock price data.

# Dependencies
The following Python libraries are used in this project:

#
```
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.arima.model import ARIMA
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
```
#
pandas: For data manipulation and analysis.

numpy: For numerical operations.

sklearn.preprocessing.MinMaxScaler: For data scaling.

statsmodels.tsa.arima.model.ARIMA: For ARIMA modeling.

datetime: For working with dates and times.

These packages are installed in your Python environment before running the code.
#

# Dataset
The code loads historical stock price data from a CSV file named World_Stock_Prices.csv. 

It contains various stock tickers along with corresponding price data (Open, Close, High, Low).

Load Data: The pd.read_csv function is used to load the dataset into a DataFrame named stock_data.

Convert Date to Datetime: The pd.to_datetime function converts the 'Date' column to a datetime format for easier manipulation.

Ensuring Numeric Data: The code ensures that stock price columns ('Open', 'High', 'Low', 'Close') are in a numeric format.

Identifying Top Stocks: The top 10 stocks with the highest closing prices are identified.

#
## Loading the dataset into the stock_data data frame
```
stock_data = pd.read_csv("World_Stock_Prices.csv")
stock_data['Date'] = pd.to_datetime(stock_data['Date'])
```
## declaring the numeric_cols df
```
# Convert relevant columns to numeric
numeric_cols = ['Open', 'High', 'Low', 'Close']
stock_data[numeric_cols] = stock_data[numeric_cols].astype(float)

# Find the top 10 stocks with the highest closing prices
latest_closes = stock_data.groupby('Ticker')['Close'].max()
top_10_stocks = latest_closes.nlargest(10)
print(top_10_stocks)
```
#


##Libraries and Dependencies for the Scaling

To run this code, the following Python libraries are required:

pandas: For data manipulation and analysis.

sklearn.preprocessing: For scaling the data.

```
pip install pandas sklearn
```

##Scaling Setup
To scale the data, two different scalers are used: MinMaxScaler and StandardScaler. 
MinMax scaling scales the data to a range (typically [0, 1]), while Standard scaling standardizes the data to have a mean of 0 and a standard deviation of 1.
#
```
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Define the columns to scale
columns_to_scale = ['Open', 'High', 'Low', 'Close', 'Volume']

# Create scaler objects
scalers = {'MinMax': MinMaxScaler(),'Standard': StandardScaler()}
```
#
A dictionary called scalers holds the scaler objects, allowing easy iteration and application of different scaling methods.

##Data Scaling

The code creates a dictionary called scaled_data to store the scaled DataFrames for comparison. It then applies each scaler to the specified columns in the stock data.

```
# Prepare a dictionary to hold scaled data for comparison
scaled_data = {}

# Apply scaling
for scaler_name, scaler in scalers.items():
    # Make a copy of the original data to preserve it for each scaling method
    scaled_df = stock_scaler.copy()
    
    # Fit and transform the data using the current scaler
    scaled_df[columns_to_scale] = scaler.fit_transform(stock_scaler[columns_to_scale])
    
    # Store the scaled DataFrame in the dictionary
    scaled_data[scaler_name] = scaled_df
```
#
Explanation of Scaling Steps

Copy the Data: Before applying any scaling, the code makes a copy of the original data to avoid modifying it directly.

Fit and Transform: The scaler's fit_transform method is used to scale the data. 
This function first calculates the scaling parameters based on the data and then applies the transformation to the specified columns.

Store Scaled Data: The scaled DataFrame is stored in the scaled_data dictionary for future reference.
#

After applying the scaling, the code prints summary statistics for the scaled data to examine how each scaling method affected the data.

```
    # Print the summary statistics of the scaled data
    print(f"Summary Statistics for {scaler_name} Scaling:")
    print(scaled_df[columns_to_scale].describe(), "\n")
```
#

# ARIMA Model Forecasting Function
#
A function named forecast_next_7_days_from_date is defined to forecast the next 7 days from a given ARIMA model and start date. The function does the following:

* Forecast for 7 Days: Uses the ARIMA model to predict the next 7 days.
* Generate Forecast Dates: Creates a list of dates for the forecast period.
* Create a DataFrame for Forecasted Data: The forecasted 'Close' prices are stored in a DataFrame with corresponding dates.
* Populate 'Open', 'High', and 'Low': The 'Open', 'High', and 'Low' values are derived from the 'Close' prices with simple assumptions.

#
```
def forecast_next_7_days_from_date(model, start_date):
    forecast = model.forecast(steps=7)
    forecast_dates = [start_date + timedelta(days=i) for i in range(1, 8)]
    forecast_df = pd.DataFrame({
        'Date': forecast_dates,
        'Close': forecast
    })
    forecast_df['Open'] = forecast_df['Close']  
    forecast_df['High'] = forecast_df['Close'] + 10 
    forecast_df['Low'] = forecast_df['Close'] - 10  
    return forecast_df
```
#

## Define the Stock Tickers

A list of top 10 stock tickers is defined to determine which stocks will be forecasted. This list includes common high-volume stocks:
```
top_10_tickers = ["CMG", "NFLX", "ADBE", "COST", "ZM", "NVDA", "MA", "TSLA", "HD", "SPOT"]
```
* The current date is obtained to serve as the starting point for forecasting:
```
current_date = datetime.now().date()
```

## Forecasting Loop

A list named all_forecasts is created to store forecast results for each stock ticker. The code then loops through each stock ticker, 
fitting an ARIMA model to the 'Close' price, and uses the forecast_next_7_days_from_date function to forecast the next 7 days. Here's a breakdown of what happens in this loop:

* Filter Data by Ticker: Only select data for the current ticker.
* Sort Data by Date: Ensure the data is sorted chronologically.
* Fit ARIMA Model: Use an ARIMA order of (7, 2, 1) to fit the model.
* Forecast for 7 Days: Forecast the next 7 days from the current date.
* Add Ticker Information: Add the stock ticker to the forecast DataFrame.
* Store the Forecast: Append the forecast DataFrame to the all_forecasts list.

```
  all_forecasts = []

for ticker in top_10_tickers:
    stock_ticker_data = stock_data[stock_data['Ticker'] == ticker]
    stock_ticker_data = stock_ticker_data.sort_values(by='Date')
    
    arima_order = (7, 2, 1)  # Example order, can be adjusted
    model = ARIMA(stock_ticker_data['Close'], order=arima_order)
    arima_fit = model.fit()
    
    forecast_df = forecast_next_7_days_from_date(arima_fit, current_date)
    forecast_df['Ticker'] = ticker
    
    all_forecasts.append(forecast_df)
```
#
After completing the loop, the code concatenates all forecasts into a single DataFrame, final_forecasts, for easy analysis:

```
final_forecasts = pd.concat(all_forecasts, ignore_index=True)
```
## Display the Final Forecast

Finally, the code prints the final forecast, displaying the results for each of the top 10 stock tickers over the 7-day forecast period:
#

```
print("7-Day Forecast for the Top 10 Stocks:")
print(final_forecasts)
```
#

### Grouping by Ticker

The code starts by creating a dictionary called ticker_forecasts to hold individual DataFrames for each stock ticker. It then groups the final_forecasts DataFrame by 'Ticker' to extract forecasts for each specific stock.

ticker_forecasts = {}
```
# Group the final forecasts by 'Ticker'
grouped_forecasts = final_forecasts.groupby('Ticker')
```
### Looping Through Groups to Store Individual DataFrames
The code loops through the grouped data to store the forecasts for each ticker in the ticker_forecasts dictionary. This allows for easy retrieval of forecasted data by stock ticker.

```
for ticker, group in grouped_forecasts:
    # Store the DataFrame for each ticker in the dictionary
    ticker_forecasts[ticker] = group
```

#
```
# Display the forecasted data for each stock ticker
for ticker, forecast_df in ticker_forecasts.items():
    print(f"Forecasted Data for {ticker}:")
    print(forecast_df)
```
## Plotting the Forecasted Data

To visualize the forecasted data, the code uses matplotlib.pyplot to plot the forecasted 'Close' prices over a 7-day period. This helps in understanding trends and comparing forecasted data among different stock tickers.

```
import matplotlib.pyplot as plt

# Plotting the forecast for each stock ticker
for ticker, forecast_df in ticker_forecasts.items():
    plt.plot(forecast_df['Date'], forecast_df['Close'], label=ticker)  # Plot 'Close' price over time
    plt.title(f"7-Day Forecast for {ticker}")  # Add a title
    plt.xlabel("Date")  # Label the x-axis
    plt.ylabel("Closing Price")  # Label the y-axis
    plt.gca().xaxis.set_tick_params(rotation=30, labelsize=10)  # Rotate x-axis labels for better readability
    plt.legend()  # Add a legend
    plt.show()  # Display the plot
```
#

* Plot 'Close' Price: Plots the 'Close' price over the forecasted period for each stock ticker.
* Plot Formatting: Adds a title, labels for the x- and y-axes, and adjusts the rotation of the x-axis labels for better readability.
* Display the Plot: Calls plt.show() to render the plot, allowing visualization of the forecasted data.
