# AIT-Final-Project-Sai-Surya-Gadiraju-Team-3

## Expolatory Data Analysis

The exploratory data analysis (EDA) performed on a dataset of world stock prices. The EDA aims to understand the dataset's structure, summarize key characteristics, and identify trends in stock prices by country and industry.

### Table of Contents
Data Loading
Initial Data Exploration
Data Conversion and Cleaning
Industry-Level Analysis
Country-Level Analysis and Visualization
Results and Observations

## Data Loading
The first step is to load the dataset and ensure it has been imported correctly. The data is loaded into a Pandas DataFrame from a CSV file named "World_Stock_Prices.csv".
```
import pandas as pd
stock_data = pd.read_csv("World_Stock_Prices.csv")
# Display the first few rows to check the data structure
stock_data.head()
```
# 

## Initial Data Exploration
Once the data is loaded, it's important to get an overview of its structure. This includes checking the data types of each column, identifying missing values, and summarizing key statistics.

```
# Display information about the DataFrame
stock_data.info()

# Summary statistics for numerical columns
stock_data.describe()
```
#

## Data Conversion and Cleaning

The 'Date' column is converted to a datetime format to enable time-based operations and sorting. 

This step ensures consistency when working with time-series data.
```
# Convert 'Date' to datetime
stock_data['Date'] = pd.to_datetime(stock_data['Date'])
# Display DataFrame information to confirm the data type conversion
stock_data.info()
```
## Industry-Level Analysis

To understand stock prices at the industry level, the data is grouped by 'Industry_Tag' and the mean 'Open' and 'Close' prices are calculated. This allows for comparisons between different industries.

```
# Calculate average 'Open' and 'Close' prices by industry
industry_analysis_open = stock_data.groupby('Industry_Tag').agg({'Open': 'mean'})
industry_analysis_close = stock_data.groupby('Industry_Tag').agg({'Close': 'mean'})

# Rename columns for clarity
industry_analysis_open.rename(columns={'Open': 'Average_Open'}, inplace=True)
industry_analysis_close.rename(columns={'Close': 'Average_Close'}, inplace=True)

# Sort industries by 'Average_Open' and 'Average_Close'
top_open_industries = industry_analysis_open.sort_values(by='Average_Open', ascending=False)
top_close_industries = industry_analysis_close.sort_values(by='Average_Close', ascending=False)
```
#
## Observations on Industry Analysis
This analysis reveals which industries tend to have higher 'Open' and 'Close' prices. 
The sorted DataFrames, top_open_industries and top_close_industries, provide this information.

```
print("Top industries by average 'Open' price:")
print(top_open_industries)

print("Top industries by average 'Close' price:")
print(top_close_industries)
```
#
## Country-Level Analysis and Visualization

To explore stock price trends by country, the data is grouped by 'Date' and 'Country' to calculate the mean 'Open' and 'Close' prices. 
This section uses Plotly to visualize trends over time. Its an interactive graph

```
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# Group the data by date and country and calculate mean 'Open' and 'Close' prices
country_mean_price = stock_data.groupby(['Date', 'Country']).agg({'Open': 'mean', 'Close': 'mean'}).reset_index()

# Create a Plotly figure for visualization
fig = go.Figure()

# Add traces for 'Open' and 'Close' prices for each country
for country in country_mean_price['Country'].unique():
    country_data = country_mean_price[country_mean_price['Country'] == country]

    # Add a trace for 'Close' prices
    fig.add_trace(go.Scatter(
        x=country_data['Date'],
        y=country_data['Close'],
        mode='lines+markers',  # Line plot with markers
        name=f"{country} - Close",
        hoverinfo='x+y+text'  # Tooltip configuration
    ))
    
    # Add a trace for 'Open' prices
    fig.add_trace(go.Scatter(
        x=country_data['Date'],
        y=country_data['Open'],
        mode='lines+markers',  # Line plot with markers
        name=f"{country} - Open",
        hoverinfo='x+y+text'  # Tooltip configuration
    ))

# Update layout with titles and axis labels
fig.update_layout(
    title="Stock Price Trends with Open/Close by Country",
    xaxis_title="Date",
    yaxis_title="Stock Price",
    legend_title="Country - Open/Close"
)

# Display the plot
fig.show()
```

## Observations on Country-Level Analysis
#
The visualization shows stock price trends over time, with separate lines for 'Open' and 'Close' prices for each country. This analysis helps to identify patterns or anomalies in stock prices across different countries.

## Results and Observations

Through this EDA, several key insights are gained:

Certain industries have higher average 'Open' and 'Close' prices.

There are variations in stock price trends among different countries, with distinct patterns over time.
#

## Stock Price Analysis with PySpark and LSTM
### This README explains the code that analyzes stock price data using PySpark, applies exploratory data analysis (EDA), transforms and scales the data, and builds a Long Short-Term Memory (LSTM) model for stock price forecasting. It also covers model training, evaluation, and visualization of the results.

## Introduction

This project analyzes stock price data and uses an LSTM model to forecast stock prices. It employs PySpark for data processing and transformation, and TensorFlow/Keras, Hyper parameter tunning for building the LSTM model. The project involves exploratory data analysis, data transformation, stock data scaling, and LSTM-based forecasting.

## Dependencies
To run this code, ensure the following Python libraries are installed:

pyspark: For Spark-based data processing.

pandas: For data manipulation.

seaborn: For data visualization.

matplotlib: For plotting.

tensorflow: For building and training LSTM models.

sklearn: For data preprocessing and additional metrics.

``
pip install pyspark pandas seaborn matplotlib tensorflow sklearn
``

### Data Loading and Exploration
The code starts by creating a Spark session and loading stock price data from a CSV file into a PySpark DataFrame. It then performs initial data exploration to understand the data's structure, including basic statistics and data types.

```
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, to_date

# Create a Spark session
spark = SparkSession.builder.appName("StockDataAnalysis").getOrCreate()

# Load the stock price data from CSV
stock_data = spark.read.csv("dbfs:/FileStore/shared_uploads/sgadira3@gmu.edu/World_Stock_Prices-2.csv", header=True, inferSchema=True)

# Display the first few rows of the DataFrame
stock_data.head()

# Show summary statistics for the data
summary_stats = stock_data.describe()
summary_stats.display()
```
## Data Transformation and Rounding
The code applies transformations to round numerical columns to a consistent format, which helps with data normalization and later processing. It also creates a scatter plot to visualize the relationship between 'Open' and 'Close' prices.

```
import pyspark.sql.functions as F
import seaborn as sns
import matplotlib.pyplot as plt

# Round 'Open', 'High', 'Low', and 'Close' columns to 2 decimal places
stock_data = stock_data.withColumn("Open", F.round(F.col("Open"), 2))
stock_data = stock_data.withColumn("High", F.round(F.col("High"), 2))
stock_data = stock_data.withColumn("Low", F.round(F.col("Low"), 2))
stock_data = stock_data.withColumn("Close", F.round(F.col("Close"), 2))

# Display the data to ensure rounding is applied correctly
stock_data.display(10)

# Create a scatter plot to visualize 'Close' vs. 'Open'
sns.scatterplot(data=stock_data.toPandas(), x='Close', y='Open')
plt.xlabel("Stock Close Price")
plt.ylabel("Stock Open Price")
plt.title("Scatter Plot of Stock Close Price vs. Open")
plt.show()
```


## Stock Data Scaling with PySpark
To ensure data consistency, MinMaxScaler is used to scale the stock price data to a range between 0 and 1. This section also introduces transformations with dense vectors and extracts components from those vectors.

### Setting Up a Spark Session
The code starts by initializing a Spark session, which is the entry point to working with Spark. The appName parameter specifies the name of the application.

```
from pyspark.sql import SparkSession
# Create a Spark session for scaling
spark = SparkSession.builder.appName("StockScaling").getOrCreate()
```

* A Spark session is necessary for interacting with Spark functionalities, including DataFrames and machine learning operations.

### Filtering Data by Date Range
The data is filtered to include only the rows between January 1, 2019, and December 31, 2023. This step helps focus the analysis on a specific time period.

```
from pyspark.sql.functions import col

# Filter data between 2019 and 2023 for analysis
stock_filtered = stock_data.filter((col("Date") >= "2019-01-01") & (col("Date") <= "2023-12-31"))
```

This step uses the filter method with conditions applied to the 'Date' column. The col function is used to refer to DataFrame columns in PySpark.


### Creating Dense Vectors for Stock Features
A user-defined function (UDF) is defined to create dense vectors from 'Open', 'Close', 'High', and 'Low' columns. Dense vectors are commonly used in machine learning operations with Spark.

```
from pyspark.sql.functions import udf
from pyspark.ml.linalg import Vectors, VectorUDT

# Create dense vectors from 'Open', 'Close', 'High', 'Low'
to_vector = udf(lambda a, b, c, d: Vectors.dense([a, b, c, d]), VectorUDT())
stock_filtered = stock_filtered.withColumn("features", to_vector("Open", "Close", "High", "Low"))

```

The to_vector UDF converts the 'Open', 'Close', 'High', and 'Low' columns into a dense vector. This is useful for applying machine learning operations, such as scaling and modeling.



### Applying MinMaxScaler to Normalize Data
MinMaxScaler is used to normalize the data by scaling it to a specified range (default is 0 to 1). This step is important for ensuring consistent data for machine learning models.

```
from pyspark.ml.feature import MinMaxScaler

# Initialize MinMaxScaler and fit to the filtered data
scaler = MinMaxScaler(inputCol="features", outputCol="scaled_features")
scaler_model = scaler.fit(stock_filtered)
scaled_stock_data = scaler_model.transform(stock_filtered)

```
The MinMaxScaler is initialized with the input and output columns specified. The scaler is then fit to the filtered data to calculate the scaling parameters. The transform method applies the scaling to create the scaled_stock_data DataFrame, which contains the normalized data.

### Extracting Components from Dense Vectors
To work with the scaled data, a UDF is used to extract individual components from the dense vector. This step creates new columns for the scaled 'Open', 'Close', 'High', and 'Low' values.

```
from pyspark.sql.functions import lit

# UDF to extract components from the dense vector
extract_element = udf(lambda v, i: float(v[i]), FloatType())

# Apply UDF to extract 'Scaled_Open', 'Scaled_Close', 'Scaled_High', and 'Scaled_Low'
scaled_stock_data = scaled_stock_data.withColumn("Scaled_Open", extract_element("scaled_features", lit(0))) \
    .withColumn("Scaled_Close", extract_element("scaled_features", lit(1))) \
    .withColumn("Scaled_High", extract_element("scaled_features", lit(2))) \
    .withColumn("Scaled_Low", extract_element("scaled_features", lit(3)))
```

The extract_element UDF takes a dense vector and an index and returns the corresponding value as a float. The lit function is used to pass a literal value to the UDF, specifying which component to extract. The withColumn method adds the new columns to the DataFrame.

### Displaying the Scaled Stock Data
The final step is to display the scaled stock data to verify that the transformations and scaling were applied correctly.

```
# Display the scaled stock data
scaled_stock_data.select("Date", "Scaled_Open", "Scaled_Close", "Scaled_High", "Scaled_Low").show(5)
```

The select method selects specific columns to display, and the show method displays the first few rows of the selected data. This is useful for verifying that the scaled data has been created and formatted correctly.


### Setting Up the Environment
First, the necessary libraries are imported, including TensorFlow for building the LSTM model and scikit-learn for data preprocessing.

```
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.regularizers import l2
from sklearn.preprocessing import MinMaxScaler
TensorFlow provides the tools to build and train neural networks, while MinMaxScaler from scikit-learn is used to normalize data.
```

### Converting Spark DataFrame to Pandas DataFrame
The code converts the Spark DataFrame (scaled_stock_data) to a Pandas DataFrame for further processing and analysis.

```
# Convert Spark DataFrame to Pandas DataFrame
pandas_df = scaled_stock_data.toPandas()
```

Converting to Pandas allows the code to use additional data manipulation and analysis tools before building the LSTM model.

### Adding a Simple Moving Average (SMA) Feature
A simple moving average with a 10-day window is added as a new feature to capture short-term trends in the stock prices. This is a common feature used in stock price analysis.

```
# Add a simple moving average (SMA) feature with a 10-day window
pandas_df['SMA_10'] = pandas_df['Close'].rolling(window=10).mean()
pandas_df.dropna(inplace=True)# Drop rows with NaN values 
```
The rolling(window=10).mean() method calculates the 10-day moving average, and dropna() removes rows with NaN values resulting from the moving average calculation.

### Creating Input Sequences for LSTM
To use LSTM, the data needs to be converted into sequences, where each sequence consists of a specified number of time steps. The code defines a function to create these sequences for training and testing the LSTM model.

```
# Function to create input sequences for LSTM
def create_sequences(data, n_steps):
    X, y = [], []
    for i in range(n_steps, len(data)):
        X.append(data[i - n_steps:i])
        y.append(data[i, :4])  # Predict Open, Close, High, Low
    return np.array(X), np.array(y)
```

n_steps specifies the number of time steps in each sequence.
X contains the input sequences, while y contains the corresponding target values.
y captures the original 'Open', 'Close', 'High', and 'Low' values, which are used as targets for the LSTM model.

### Selecting and Scaling Features for LSTM
The code selects the features to be used for LSTM and scales them using MinMaxScaler to ensure consistent input data. Scaling normalizes the data to a specific range (typically 0 to 1).

```
# Select and scale features and targets for LSTM
features = ['Scaled_Open', 'Scaled_Close', 'Scaled_High', 'Scaled_Low', 'SMA_10']
targets = ['Open', 'Close', 'High', 'Low']
all_columns = features + targets

scaler = MinMaxScaler(feature_range=(0, 1))
data = scaler.fit_transform(pandas_df[all_columns])
```
The selected features include the scaled 'Open', 'Close', 'High', and 'Low' prices, along with the 'SMA_10'.
The targets are the original 'Open', 'Close', 'High', and 'Low' values.
The fit_transform() method scales the data based on the defined feature range.

### Creating Sequences for LSTM
After scaling the data, the code creates sequences with a specified number of time steps. These sequences are used to train and test the LSTM model.

```
# Create sequences with a specified number of steps
n_steps = 5
X, y = create_sequences(data, n_steps)
```
n_steps is set to 5, meaning each sequence consists of 5 time steps.
X contains the input sequences, and y contains the corresponding target values.

### Splitting the Data into Training and Testing Sets
The code splits the sequences into training and testing sets, typically using an 80/20 split, to train and evaluate the LSTM model.

```
# Split the data into training and testing sets
train_size = int(0.8 * len(X))
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]
```
train_size determines the number of sequences to be used for training.
X_train and y_train contain the training data.
X_test and y_test contain the testing data.

### Building and Training the LSTM Model
This section builds the LSTM model with regularization and dropout to prevent overfitting. The architecture includes two LSTM layers with different configurations.

```
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.regularizers import l2

# Build an LSTM model with regularization and dropout
model = Sequential([
    LSTM(100, return_sequences=True, activation='relu', input_shape=(n_steps, X.shape[2]), kernel_regularizer=l2(0.01)),
    Dropout(0.3),
    LSTM(50, activation='relu', kernel_regularizer=l2(0.01)),
    Dropout(0.3),
    Dense(4)  # Predict Open, Close, High, Low
])

# Compile the model with Adam optimizer and Mean Squared Error (MSE) loss
model.compile(optimizer='adam', loss='mse')
```

The LSTM model has two LSTM layers: the first with 100 units, returning sequences, and the second with 50 units.
Dropout is used after each LSTM layer to reduce overfitting.
The model ends with a Dense layer with 4 outputs (predicting 'Open', 'Close', 'High', and 'Low').
The model is compiled with the Adam optimizer and Mean Squared Error (MSE) loss function.
The code trains the LSTM model using the training data and includes a validation split for monitoring during training.

```
# Train the LSTM model with a validation split for monitoring
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.1, verbose=1)

```
epochs=50 means the model is trained for 50 epochs.
batch_size=32 specifies the batch size for training.
validation_split=0.1 uses 10% of the training data for validation to monitor training progress.
verbose=1 provides detailed training output.



#
This README file provides an explanation of the code used to forecast stock prices for specific tickers using the ARIMA (AutoRegressive Integrated Moving Average) model. The project imports historical stock price data, scales the data, fits ARIMA models to predict future prices, and saves the forecast results to a CSV file.


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
