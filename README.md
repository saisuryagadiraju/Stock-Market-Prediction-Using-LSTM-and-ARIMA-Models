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

### Model Evaluation on Test Data
Model evaluation is the process of testing a trained model on unseen data to assess its performance. In this case, the code evaluates the LSTM model using the test dataset and calculates MSE and RMSE.

```
# Evaluate the model on the test data
test_mse = model.evaluate(X_test, y_test, verbose=1)

```
The model.evaluate() function runs the model on the test data (X_test) and compares its predictions with the true values (y_test) to compute the loss, which is MSE in this case.
verbose=1 provides output on the evaluation process, indicating the progress and the final result.

##### Root Mean Squared Error (RMSE)
RMSE is derived from MSE by taking its square root. It is useful because it is in the same unit as the target variable, making it easier to interpret.

```
# Calculate RMSE from MSE
test_rmse = np.sqrt(test_mse)

print("Test MSE:", test_mse)
print("Test RMSE:", test_rmse) 
```
This step displays the test MSE and RMSE, giving an indication of the model's accuracy on the test dataset.
A lower RMSE generally indicates better model performance, with smaller errors in predictions.
#

### Generating Predictions
After training the LSTM model, predictions are generated using the test dataset. This step evaluates how well the model can forecast unseen data.

```
predictions = model.predict(X_test)
```

The model.predict() method is used to generate predictions from the LSTM model, using the test input data X_test.
The output predictions contains the model's forecasts for the test set.

Reshaping the Data
The code reshapes the actual and predicted data to extract specific components, allowing for detailed evaluation.

```
# Reshape to extract specific components
actual_prices = y_test.reshape(-1, 4)[:, 0]  
predicted_prices = predictions.reshape(-1, 4)[:, 0]
```
y_test.reshape(-1, 4)[:, 0] reshapes y_test to get the first component (often representing 'Open' price). The same operation is applied to predictions.
Reshaping is useful when the data is not in the desired format for comparison or analysis.

Computing Mean Absolute Error (MAE)
MAE is a common metric used to evaluate model accuracy. It measures the average absolute difference between predicted and actual values, providing an intuitive measure of prediction error.

```
from sklearn.metrics import mean_absolute_error
# Calculate Mean Absolute Error (MAE)
mae = mean_absolute_error(actual_prices, predicted_prices)
```

![image](https://github.com/saisuryagadiraju/AIT-Final-Project-Sai-Surya-Gadiraju/assets/155181311/750ca4a4-2b9b-4996-9811-4f017d921950)

MAE is calculated using mean_absolute_error(actual_prices, predicted_prices).
MAE is interpreted as the average absolute difference between predicted and actual values. A lower MAE indicates better model accuracy.

The first step is to train the LSTM model using the training data (X_train, y_train). The validation_split parameter allows monitoring the model's performance on a subset of the training data not used for actual training, which provides a measure of how well the model generalizes to unseen data.

python
Download
Copy code
history = model.fit(X_train, y_train, epochs=10, batch_size=50, validation_split=0.1, verbose=1)
epochs=10: Specifies that the model will be trained for 10 epochs (iterations through the entire training dataset).
batch_size=50: Defines the number of samples processed before updating the model weights.
validation_split=0.1: Uses 10% of the training data for validation to monitor the model's performance during training.
verbose=1: Controls the amount of information displayed during training (1 provides detailed output).
The model.fit() method returns a history object that contains information about the training process, including the loss for each epoch.

#### Extracting Training and Validation Loss
After training, the code extracts the training loss and validation loss from the history object. This data is used to plot the loss over epochs to evaluate the training process.

```
loss = history.history['loss']
val_loss = history.history['val_loss']
```
loss: The training loss for each epoch.
val_loss: The validation loss for each epoch.
Loss typically refers to the value of the loss function (in this case, Mean Squared Error), which is used to measure how well the model is learning.

### Hyperparameter Tuning with a Parameter Grid
The code defines a grid of hyperparameters to iterate over, allowing you to find the optimal combination for the LSTM model.

```
param_grid = {
    'lstm_units': [50, 100],
    'dropout_rate': [0.2, 0.3, 0.4],
    'batch_size': [32, 64]
}
```
The grid includes possible values for:
lstm_units: Number of units in the LSTM layer.
dropout_rate: Fraction of the units to drop during training (prevents overfitting).
batch_size: Number of samples processed at a time during training.


### Creating an LSTM Model with Hyperparameters
A function is defined to create an LSTM model using the specified hyperparameters.

```
def create_lstm_model(lstm_units, dropout_rate):
    model = Sequential([
        LSTM(lstm_units, return_sequences=True, activation='relu', input_shape=(n_steps, X_train.shape[2])),
        Dropout(dropout_rate),
        LSTM(lstm_units // 2, activation='relu'),
        Dropout(dropout_rate),
        Dense(4)  # Output for Open, Close, High, Low
    ])
    model.compile(optimizer='adam', loss='mse')
    return model
```
The LSTM model has two layers, with the first layer using the specified lstm_units and the second layer having half as many units.
The Dropout layers use the specified dropout_rate to reduce overfitting.
The Dense layer outputs 4 values, corresponding to Open, Close, High, and Low stock prices.
The model is compiled with the Adam optimizer and MSE loss function.

### Iterating Over Hyperparameters and Training the Model
The code iterates over the hyperparameter grid and trains an LSTM model for each combination of lstm_units, dropout_rate, and batch_size.

```
results = []
for lstm_units in param_grid['lstm_units']:
    for dropout_rate in param_grid['dropout_rate']:
        model = create_lstm_model(lstm_units, dropout_rate)
        model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.1, verbose=1)
        mse = model.evaluate(X_test, y_test)
        results.append({'lstm_units': lstm_units, 'dropout_rate': dropout_rate, 'mse': mse})
```
The code trains each LSTM model for 10 epochs, with a validation split to monitor overfitting.
After training, the model is evaluated using the test data to obtain the MSE.
The results, including lstm_units, dropout_rate, and the computed mse, are stored in a list for comparison.

### Finding the Best Hyperparameters
A custom function is used to find the best hyperparameters by comparing the MSE of each model. The minimum MSE is identified to determine the optimal combination of hyperparameters.

```
def custom_min(results):
    if not results:
        return None
    min_result = results[0]
    for result in results[1:]:
        if result['mse'] < min_result['mse']:
            min_result = result
    return min_result

best_result = custom_min(results)
print("Best Hyperparameters:", best_result)
```
![image](https://github.com/saisuryagadiraju/AIT-Final-Project-Sai-Surya-Gadiraju/assets/155181311/e13668f8-1cf3-48a4-aa29-cf1a26cf2838)
#
custom_min() iterates over the results list to find the entry with the lowest MSE.
The best result is displayed to identify the optimal hyperparameters.
### Retraining the Best Model
Using the best hyperparameters, the code retrains the LSTM model and evaluates it on the test data.

```
best_lstm_units = best_result['lstm_units']
best_dropout_rate = best_result['dropout_rate']

best_model = create_lstm_model(best_lstm_units, best_dropout_rate)

best_model.fit(X, y, epochs=10, batch_size=50, verbose=1)

test_loss = best_model.evaluate(X_test, y_test)
print("Test Loss:", test_loss)
```
The best model is retrained using the entire dataset (X, y) for the specified number of epochs and batch size.
The test loss is evaluated to check how well the model performs with the best hyperparameters.

### Additional Metrics and Predictions
Finally, the code generates predictions and calculates additional metrics like MSE, RMSE, and MAE to assess the model's accuracy.

```
predictions = best_model.predict(X_test)

from sklearn.metrics import mean_squared_error, mean_absolute_error

# Calculate MSE, RMSE, MAE
mse = mean_squared_error(y_test, predictions)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, predictions)

print("Mean Squared Error (MSE):", mse)
print("Root Mean Squared Error (RMSE):", rmse)
print("Mean Absolute Error (MAE):", mae)
```
The predictions are generated from the best model using the test data.
mean_squared_error calculates MSE between the actual and predicted values.
np.sqrt(mse) calculates RMSE, providing a more interpretable measure of error.
mean_absolute_error computes MAE, indicating the average absolute error in the predictions.
The results for MSE, RMSE, and MAE are displayed to evaluate the model's accuracy.

![image](https://github.com/saisuryagadiraju/AIT-Final-Project-Sai-Surya-Gadiraju/assets/155181311/f1e214a8-1802-4990-af5e-673a956902e6)

#

This LSTM Predicton Analysis was done in Jupyter instead of Databricks to avoid the complexity to store the files in the local host (local computer)
#

### Creating Multivariable Datasets
The code creates a function to generate training sequences for LSTM. It defines a look-back period and a number of future days to predict.

```
def create_multivariable_dataset(data, look_back=100, future_days=7):
    X, Y = [], []
    for i in range(len(data) - look_back - future_days + 1):
        X.append(data[i:(i + look_back), :])
        Y.append(data[(i + look_back):(i + look_back + future_days), :])
    return np.array(X), np.array(Y)
```
Look-back Period: The number of past days used to predict the future (100 in this case).
Future Days: The number of future days to forecast (7 in this case).

### Preparing Datasets for Each Ticker
The code loops through each unique stock ticker, normalizes the features, and creates the training datasets (X and y).

```
# Prepare the dataset for each ticker
for ticker in unique_tickers:
    print(f"Processing data for {ticker}...")
    data = stock_data[stock_data['Ticker'] == ticker]
    data = data.sort_values('Date')
    
    features = data[['Open', 'High', 'Low', 'Close']].values
    
    # Normalize the features
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_features = scaler.fit_transform(features)
    scalers[ticker] = scaler
    
    X, y = create_multivariable_dataset(scaled_features)
    X = np.reshape(X, (X.shape[0], X.shape[1], 4))  # 4 features: Open, High, Low, Close
    
    np.save(f"{ticker}_X.npy", X)
    np.save(f"{ticker}_y.npy", y)
```
Normalize Features: Scales the features to a consistent range (0 to 1).

Create Training Sequences: Calls create_multivariable_dataset() to generate sequences for training.

Save Datasets: Saves the generated X and y datasets to files for later use.
### Model Building and Training

The code defines a function to build the LSTM model and then trains a separate model for each stock ticker using the saved datasets.

Building the LSTM Model
The model is built with two LSTM layers, dropout to prevent overfitting, and a dense layer for the output.

```
def build_model(input_shape):
    model = Sequential([
        LSTM(15, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(7 * 4),  
        Reshape((7, 4))  
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model
````
LSTM Layers: The first LSTM layer has 15 units with return_sequences=True, and the second has 50 units.

Dropout: Dropout is applied after each LSTM layer to reduce overfitting.
Dense Layer: The dense layer outputs predictions for 7 days, each with 4 features (Open, High, Low, Close).

Reshape Layer: This reshapes the output into the desired format.

Model Compilation: The model is compiled with the Adam optimizer and Mean Squared Error (MSE) loss function.

### Training the Model for Each Ticker

The code loops through each unique stock ticker, builds the LSTM model, and trains it with specified hyperparameters.

```
for ticker in unique_tickers:
    print(f"Training model for {ticker}...")
    X = np.load(f"{ticker}_X.npy")
    y = np.load(f"{ticker}_y.npy")
    
    model = build_model((X.shape[1], X.shape[2]))
    
    checkpoint_filepath = f'{ticker}_best_model.keras'  # Make sure to use '.keras' or '.h5' as needed
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    model_checkpoint = ModelCheckpoint(checkpoint_filepath, monitor='val_loss', save_best_only=True)
    
    model.fit(X, y, epochs=10, batch_size=50, validation_split=0.2, callbacks=[early_stopping, model_checkpoint])
```

Training the Model: The code trains the model for each stock ticker using the corresponding datasets (X, y).

Validation Split: A 20% validation split allows monitoring during training to detect overfitting.

Callbacks: The code uses EarlyStopping to halt training if the validation loss doesn't improve for 10 epochs and ModelCheckpoint to save the best model based on validation loss.
Training Parameters: The model is trained for 10 epochs with a batch size of 50.

### Callbacks for Model Training
The code uses callbacks to improve the training process by adding mechanisms to stop training early if necessary and save the best model.

EarlyStopping
EarlyStopping stops training if there's no improvement in the validation loss within a specified number of epochs.

```
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)


Monitor: EarlyStopping monitors the 'val_loss'.

Patience: Training will stop if there's no improvement for 10 epochs.

Restore Best Weights: Restores the best model weights if training stops early.

### ModelCheckpoint
ModelCheckpoint saves the best model based on the monitored metric.

```
model_checkpoint = ModelCheckpoint(
    f'{ticker}_best_model.keras', 
    monitor='val_loss', 
    save_best_only=True
)
Filepath: The file path to save the best model.
Monitor: The monitored metric is 'val_loss'.
Save Best Only: Saves only the best model based on validation loss.

```
# Save the final model
model.save(f'{ticker}_final_model.keras')
```
model.save(): This method saves the entire Keras model, including the architecture, weights, and optimizer state. The saved model can be loaded later for inference or continued training.

f'{ticker}_final_model.keras': This is the file path where the model is saved. It uses f-string formatting to include the ticker's name in the file name, ensuring that each unique stock ticker has a corresponding model file.

To load the saved model 
```
from tensorflow.keras.models import load_model

# Load the saved model
loaded_model = load_model(f'{ticker}_final_model.keras')
```

### Splitting Data into Training and Testing Sets
To train and evaluate the model, the data is split into training and testing sets. This allows you to train the model on one set and evaluate it on a separate set to gauge its performance.

````
from sklearn.model_selection import train_test_split

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

train_test_split(X, y, test_size=0.2, random_state=42): Splits the dataset into training and testing sets. The test_size=0.2 parameter means 20% of the data is used for testing, and 80% for training. The random_state=42 ensures reproducibility, allowing you to get the same split every time.

X_train, X_test, y_train, y_test: These represent the training and testing datasets for input (X) and output (y).

#### Saving the Datasets
After splitting the data, the code saves the training and testing sets to files for later use. This is useful for reproducibility and sharing datasets among team members or across different environments.
```
import numpy as np

# Save the datasets
np.save(f"data/{ticker}_X_train.npy", X_train)
np.save(f"data/{ticker}_X_test.npy", X_test)
np.save(f"data/{ticker}_y_train.npy", y_train)
np.save(f"data/{ticker}_y_test.npy", y_test)
```
np.save(filename, array): Saves the numpy array to a file. This allows you to easily reload the dataset later for training or evaluation.

f"data/{ticker}_X_train.npy": The file path for the training dataset for the specific stock ticker. Using f-strings, the code incorporates the stock ticker name into the file path, ensuring unique file names for each ticker.

Saving all datasets: The code saves X_train, X_test, y_train, and y_test to separate files, providing the full set of training and testing data for each stock ticker.


### Evaluating Training Loss Before Further Training
The code evaluates the training loss of the loaded model on the training dataset before any additional training. This is useful to understand the model's initial state and performance.

```
# Evaluate the model on training data to get the training loss
train_loss = model.evaluate(X_train, y_train, verbose=1)
```
model.evaluate(X_train, y_train, verbose=1): This method evaluates the model's performance on the training dataset, providing the loss (typically Mean Squared Error). The verbose=1 parameter controls the output detail during evaluation.

The evaluation gives a baseline measure of the model's training loss before any additional training or fine-tuning.

### Continuing Training or Fine-Tuning
The code continues training or fine-tunes the model on the training dataset. This is useful for improving the model's accuracy or adapting it to new data.

```
# Continue training the model or fine-tuning
history = model.fit(X_train, y_train, epochs=30, validation_split=0.2, batch_size=50, verbose=1)
```
model.fit(): This method continues training the model on the training dataset.
epochs=30: Specifies the number of additional training epochs.
validation_split=0.2: Uses 20% of the training data for validation to monitor overfitting.
batch_size=50: Defines the batch size for training.

history: Stores the training history, which includes metrics like loss and validation loss for each epoch.


### Plotting Training and Validation Loss
To visualize the training process and detect potential overfitting, the code plots the training and validation loss over the epochs.

```
import matplotlib.pyplot as plt

# Plot the training and validation loss
plt.figure(figsize=(12, 5))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
```

plt.figure(figsize=(12, 5)): Creates a new plot with specified dimensions.
plt.plot(history.history['loss'], label='Training Loss'): Plots the training loss for each epoch.
plt.plot(history.history['val_loss'], label='Validation Loss'): Plots the validation loss for each epoch.

### Displaying the Initial Training Loss
After plotting, the code prints the training loss evaluated before the additional training or fine-tuning. This provides a baseline measure of the model's initial state.

```
# Print the training loss evaluated before the additional training
print(f"Training Loss before additional training: {train_loss}")
```

### Generating Predictions
The code starts by generating predictions from the LSTM model using the test dataset (X_test). The predict method provides the model's forecast for each input sequence.

```
# Generate predictions from the model
predictions = model.predict(X_test)
```
model.predict(X_test): This method generates predictions for the test dataset. The output predictions is an array of predicted values.

The predictions correspond to the output sequences generated by the LSTM model, typically forecasting multiple future days based on the input data.

### Reshaping Predictions
To align the predictions with the shape of the test dataset, the code reshapes the predictions array to match y_test. This is necessary for proper comparison and plotting.

```
# Reshape predictions to match the shape of the test dataset
predictions = predictions.reshape(y_test.shape)

```
predictions.reshape(y_test.shape): This reshapes the predictions array to have the same shape as y_test. This step ensures that the predictions and the test data have the same format, allowing for accurate comparison.

### Extracting Specific Components
To plot the actual and predicted prices, the code extracts specific components from the reshaped arrays. This example focuses on the 'Close' price, which is usually the primary target for stock price prediction.

```
# Extract the 'Close' prices from the first day of the sequences
actual = y_test[:, 0, 3]  # 'Close' price of the first day in the output sequence
predicted = predictions[:, 0, 3]
```
y_test[:, 0, 3]: This selects the 'Close' price from the first day of the output sequence in the test dataset.
predictions[:, 0, 3]: This extracts the corresponding 'Close' price from the predictions.
By focusing on the 'Close' price, the code aligns the actual and predicted prices for comparison.

### Plotting Actual vs. Predicted Prices
The code plots the actual and predicted prices to visualize how well the model's predictions align with the actual stock prices.

```
import matplotlib.pyplot as plt

# Create a new plot to visualize actual vs. predicted prices
plt.figure(figsize=(15, 7))
plt.plot(actual, label='Actual Prices')  # Plot the actual 'Close' prices
plt.plot(predicted, label='Predicted Prices')  # Plot the predicted 'Close' prices
plt.title('Actual vs. Predicted Prices')  # Set the plot title
plt.xlabel('Time (Days)')  # Label the x-axis
plt.ylabel('Price')  # Label the y-axis
plt.legend()  # Add a legend to identify the lines
plt.show()  # Display the plot
```

![image](https://github.com/saisuryagadiraju/AIT-Final-Project-Sai-Surya-Gadiraju/assets/155181311/09b987a6-7c1e-4231-9129-ed2de2873b7f)


### Setting Up Model Checkpoints
Before training, the code creates a ModelCheckpoint callback to save the best model based on a specific metric. This helps ensure the best model is retained during training.

````
from tensorflow.keras.callbacks import ModelCheckpoint

# Define the model directory
model_dir = "path/to/models/"  # Adjust to your preferred path

# Iterate through each unique stock ticker
for ticker in unique_tickers:
    # Build a new model for each ticker
    model = build_model((X_train.shape[1], X_train.shape[2]))
    
    # Define the checkpoint file path for the best model
    checkpoint_filepath = f'{model_dir}{ticker}_best_model.keras'  # Adjust the path as needed
    
    # Create a ModelCheckpoint to save the best model
    model_checkpoint = ModelCheckpoint(checkpoint_filepath, monitor='val_loss', save_best_only=True)
```
#

### Training the Model
The code trains the model for each stock ticker, using a validation split to monitor the training process and improve generalization.
```
    # Train the model
    model.fit(X_train, y_train, epochs=10, batch_size=50, validation_split=0.2, callbacks=[model_checkpoint])
```
model.fit(): Trains the model with specified parameters.
epochs=10: The number of training epochs.
batch_size=50: Defines the batch size for training.
validation_split=0.2: Uses 20% of the training data for validation, allowing you to monitor overfitting.
callbacks=[model_checkpoint]: Includes the ModelCheckpoint to save the best model based on validation loss.

#

### Saving the Final Model
After training, the code saves the final model to a specified path. This ensures that the model is available for later use, such as inference, deployment, or further fine-tuning.

```
    # Save the final model
    final_model_path = f'{model_dir}{ticker}_final_model.keras'
    model.save(final_model_path)  # Save the trained model
    print(f"Model saved at {final_model_path}")  
```

final_model_path: Defines the file path for the final model, ensuring it includes the stock ticker's name for uniqueness.

model.save(final_model_path): Saves the trained model to the specified file path. This includes the model's architecture, weights, and optimizer state.
#

### Loading the Trained Model
The code starts by loading a pre-trained model from a specified file path. This is usually done to make predictions or continue training.

```
from tensorflow.keras.models import load_model
# Load the trained model
model = load_model(f'models/{ticker}_best_model.keras')
```

load_model(): This function loads a saved Keras model, allowing you to use it for inference or further training.

f'models/{ticker}_best_model.keras': The file path to the saved model. Using f-string formatting, the code references the correct model for the given stock ticker.
#

### Predicting with the LSTM Model
After loading the model, the code predicts stock prices using the prepared input data. This step generates the model's output based on the input sequences.

```
# Predict using the prepared input data
predicted = model.predict(input_data)
```

model.predict(input_data): Generates predictions from the input data. This produces an array of predictions based on the trained model's architecture and learned weights.
The input_data variable represents the input sequences for the LSTM model. These should be prepared in a way that matches the model's expected input shape and structure.
#
### Transforming Predictions to Original Scale
The predicted values are typically normalized or scaled to a consistent range during preprocessing. To understand the predictions in their original context, they must be transformed back to the original scale using the appropriate scaler.

```
# Inverse transform the predicted values to the original scale
predicted_prices = scalers[ticker].inverse_transform(predicted.reshape(-1, 4)).reshape(predicted.shape)

```
scalers[ticker]: This is the specific scaler used to normalize the data during preprocessing. By referencing the correct scaler, the code can accurately transform the predictions back to their original scale.
predicted.reshape(-1, 4): This reshapes the predictions to a two-dimensional format, suitable for inverse transformation. The -1 keeps the number of rows consistent, while 4 represents the number of features (Open, High, Low, Close).


scalers[ticker].inverse_transform(): This method reverses the scaling, transforming the predictions back to the original scale.

reshape(predicted.shape): Reshapes the predictions to their original three-dimensional format after the inverse transformation.

### Creating a DataFrame for All Predictions
An empty list is created to store predictions from all unique stock tickers. This will be converted to a DataFrame at the end for easier handling and visualization.


```
# Create a DataFrame to store all predictions
all_predictions = []

```
#

### Looping Over Unique Tickers
The code loops over each unique stock ticker, loads the corresponding model, and generates predictions.

```
for ticker in unique_tickers:
    # Define the path to the best model
    model_path = f'models/{ticker}_best_model.keras'

    # Check if the model file exists
    if not os.path.isfile(model_path):
        print(f"No model found for {ticker}, skipping.")
        continue

    # Load the saved model
    model = load_model(model_path)
```
model_path: The file path to the saved model for the current stock ticker. This should point to the best model obtained during training.
os.path.isfile(model_path): Checks if the model file exists. If it doesn't, the loop skips this ticker to avoid errors.
load_model(): Loads the model from the specified file path.

### Preparing Input Data
To generate predictions, the code prepares the input data for the LSTM model. This typically involves creating a dataset with the appropriate look-back period.

```
#### Prepare input data for the model
input_data = prepare_input_data(ticker, stock_data, look_back=100) 
```
prepare_input_data(ticker, stock_data, look_back=100): This function (not defined in the snippet) is assumed to create the required input data for the model, with a specific look-back period.
look_back=100: Specifies the number of past days used to generate predictions. Ensure this matches the configuration used during model training.
Predicting Future Prices
Using the prepared input data, the code predicts future stock prices for the specified stock ticker. The predictions are then transformed back to the original scale using the appropriate scaler.

```
# Predict future prices
predicted = model.predict(input_data)
```

#### Inverse transform the predictions to the original scale
```
predicted_prices = scalers[ticker].inverse_transform(predicted.reshape(-1, 4)).reshape(predicted.shape)
```
model.predict(input_data): Generates predictions from the LSTM model using the prepared input data.
scalers[ticker].inverse_transform(): This step reverses the scaling applied during preprocessing, transforming the predictions back to the original scale. This ensures the predictions are in the same units as the original data.
reshape(): Reshapes the predictions to match the expected output format.
#

#### Generating Future Dates
The code creates a range of future dates to associate with the predicted prices. This allows you to assign a date to each prediction.
```
# Generate future dates
today = pd.Timestamp('today').normalize()
date_range = pd.date_range(start=today, periods=7, freq='D')
pd.Timestamp('today').normalize(): Creates a timestamp for today's date, normalized to remove the time component.
pd.date_range(start=today, periods=7, freq='D'): Creates a range of 7 days starting from today, with a frequency of one day ('D').
```

#### Storing Predictions in a DataFrame
The code creates a list of prediction dictionaries, associating each predicted price with the corresponding stock ticker and date. This list is then converted to a DataFrame for easier handling and visualization.

```
# Prepare and store predictions
for i in range(7):
    all_predictions.append({
        'Date': date_range[i],
        'Ticker': ticker,
        'Open': predicted_prices[0, i, 0],
        'Low': predicted_prices[0, i, 1],
        'High': predicted_prices[0, i, 2],
        'Close': predicted_prices[0, i, 3]
    })
```
# 
Convert the list of predictions to a DataFrame
predictions_df = pd.DataFrame(all_predictions)
```
all_predictions.append(): Adds a new dictionary to the list of predictions, containing the date, stock ticker, and predicted 'Open', 'Low', 'High', and 'Close' prices for the first day of the forecast period.
pd.DataFrame(all_predictions): Converts the list of predictions into a DataFrame, allowing for easier handling, analysis, or export.
```
Displaying the Predictions DataFrame
Finally, the code displays the DataFrame containing all the predictions to provide a comprehensive view of the results.

#


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

A list named all_forecasts is created to store forecast results for each stock ticker.

The code then loops through each stock ticker,

fitting an ARIMA model to the 'Close' price, and uses the forecast_next_7_days_from_date function to forecast the next 7 days.

Here's a breakdown of what happens in this loop:

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
* Plot 'Close' Price: Plots the 'Close' price over the forecasted period for each stock ticker.
* Plot Formatting: Adds a title, labels for the x- and y-axes, and adjusts the rotation of the x-axis labels for better readability.
* Display the Plot: Calls plt.show() to render the plot, allowing visualization of the forecasted data.


#
#

## SVM Model for Stock Price Prediction

This Python script utilizes Support Vector Machine (SVM) models to predict stock price movements and analyze stock data.
It employs the `yfinance` library to fetch historical stock data and `scikit-learn` for SVM model training and evaluation.

### Prerequisites

 Python 3.x
 Libraries: pandas, yfinance, scikit-learn, matplotlib

### Installation

1. Ensure Python 3.x is installed.
2. Install required libraries using pip:

```
 pip install pandas yfinance scikit-learn matplotlib
```

## Usage

*Import Required Libraries

Import necessary Python libraries including pandas, yfinance, scikit-learn, and matplotlib.

2. Define Top Company Symbols

Specify a list of top company symbols for which the SVM model will be applied.

3. Calculate Strategy Returns

Function: `calculate_strategy_returns(symbol)

Fetch historical stock data for the given symbol.

Perform feature engineering by calculating price differences and binary labeling of price movements.

Split the data into training and testing sets.

Train an SVM model using the training data.

Evaluate the model's accuracy, print classification report, and plot confusion matrix.

4. Predict Next Day Price Movement

Function: `predict_next_day_price(symbol)

Fetch historical stock data for the given symbol.

Feature engineering and data preparation similar to the previous step.

Train an SVM model using all available data.

Predict the price movement for the next day based on the latest data point.

Print the prediction whether the price will go up or down.

5. Plot ROC Curve and Precision-Recall Curve

* Function: `plot_roc_and_pr_curves(model, X_test, y_test)`
* Plot Receiver Operating Characteristic (ROC) curve and Precision-Recall curve for model evaluation.
* Compute and visualize the area under the curves.

6. Plot Feature Importance

*  Function: `plot_feature_importance(X, model)`
* Plot the importance of features using permutation importance.
* Visualize feature importance scores for the SVM model.

#### Example Usage

1. Specify the top company symbols in the `top_companies` list.

2. Run the script to train SVM models, evaluate performance, and visualize results.


python svm_stock_prediction.py
import pandas as pd
stock_data=pd.read_csv("C:/Users/HP/Downloads/World_Stock_Prices 1.csv")
stock_data.head()
stock_data.info()


    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 279753 entries, 0 to 279752
    Data columns (total 12 columns):
     #   Column        Non-Null Count   Dtype  
    ---  ------        --------------   -----  
     0   Date          279753 non-null  object 
     1   Open          279753 non-null  float64
     2   High          279753 non-null  float64
     3   Low           279753 non-null  float64
     4   Close         279753 non-null  float64
     5   Volume        279753 non-null  int64  
     6   Dividends     279753 non-null  float64
     7   Stock Splits  279753 non-null  float64
     8   Brand_Name    279753 non-null  object 
     9   Ticker        279753 non-null  object 
     10  Industry_Tag  279753 non-null  object 
     11  Country       279753 non-null  object 
    dtypes: float64(6), int64(1), object(5)
    memory usage: 25.6+ MB


stock_data.describe()


```
##SVM MODEL
import pandas as pd
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_curve, auc, precision_recall_curve
```

#### List of top company symbols
```
top_companies = ['AAPL', 'NKE', 'COST', 'AMZN']
```

# Function to calculate and plot strategy returns for a given symbol
def calculate_strategy_returns(symbol):
    # Fetch historical stock data
    stock_data = yf.download(symbol, start='2019-01-01', end='2023-09-20')

    # Feature Engineering
    stock_data['Price_Diff'] = stock_data['Close'].diff()
    stock_data['Price_Up'] = (stock_data['Price_Diff'] > 0).astype(int)

    # Drop NaN values and unnecessary columns
    stock_data.dropna(inplace=True)
    stock_data.drop(['Price_Diff', 'Adj Close', 'Volume'], axis=1, inplace=True)

    # Define features and target variable
    X = stock_data.drop('Price_Up', axis=1)
    y = stock_data['Price_Up']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the Support Vector Machine (SVM) model
    model = SVC(kernel='linear', random_state=42)  # Try different kernels ('poly', 'rbf', etc.)
    model.fit(X_train, y_train)

    # Make predictions on the testing set
    y_pred = model.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy for {symbol}: {accuracy}")

    # Print classification report
    print(f"Classification Report for {symbol}:")
    print(classification_report(y_test, y_pred))

    # Plot confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f"Confusion Matrix for {symbol}")
    plt.colorbar()
    plt.xticks([0, 1], ['Predicted Down', 'Predicted Up'], rotation=45)
    plt.yticks([0, 1], ['Actual Down', 'Actual Up'])
    plt.xlabel('True label')
    plt.ylabel('Predicted label')
    plt.show()
```
#
####  Loop through top company symbols and calculate strategy returns for each
```
for company_symbol in top_companies:
    calculate_strategy_returns(company_symbol)
```
#### Function to predict the price movement for the next day
```
def predict_next_day_price(symbol):
    # Fetch historical stock data
    stock_data = yf.download(symbol, start='2019-01-01', end='2023-09-20')

    # Feature Engineering
    stock_data['Price_Diff'] = stock_data['Close'].diff()
    stock_data['Price_Up'] = (stock_data['Price_Diff'] > 0).astype(int)

    # Drop NaN values and unnecessary columns
    stock_data.dropna(inplace=True)
    stock_data.drop(['Price_Diff', 'Adj Close', 'Volume'], axis=1, inplace=True)

    # Define features and target variable
    X = stock_data.drop('Price_Up', axis=1)
    y = stock_data['Price_Up']

    # Train the Support Vector Machine (SVM) model using all data
    model = SVC(kernel='linear', random_state=42)
    model.fit(X, y)

    # Extract features for the most recent data point (last day in the dataset)
    last_day_features = X.iloc[-1].values.reshape(1, -1)

    # Predict the price movement for the next day
    next_day_prediction = model.predict(last_day_features)[0]

    # Print the prediction
    if next_day_prediction == 1:
        print(f"For {symbol}: Predicted price will go up tomorrow.")
    else:
        print(f"For {symbol}: Predicted price will go down tomorrow.")
```
#### Loop through top company symbols and predict the price movement for the next day
```
for company_symbol in top_companies:
    predict_next_day_price(company_symbol)
```

#### Function to plot ROC Curve and Precision-Recall Curve
```
def plot_roc_and_pr_curves(model, X_test, y_test):
    # Predict probabilities
    if isinstance(model, CalibratedClassifierCV):
        y_scores = model.predict_proba(X_test)[:, 1]
    else:
        y_scores = model.decision_function(X_test)

    # Compute ROC curve and ROC area
    fpr, tpr, _ = roc_curve(y_test, y_scores)
    roc_auc = auc(fpr, tpr)

    # Compute Precision-Recall curve and area under the curve
    precision, recall, _ = precision_recall_curve(y_test, y_scores)
    pr_auc = auc(recall, precision)

    # Plot ROC Curve
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")

    # Plot Precision-Recall Curve
    plt.subplot(1, 2, 2)
    plt.plot(recall, precision, color='red', lw=2, label='Precision-Recall curve (area = %0.2f)' % pr_auc)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")

    plt.tight_layout()
    plt.show()
```

#### Loop through top company symbols and plot ROC Curve and Precision-Recall Curve
for company_symbol in top_companies:
    print("=====", company_symbol, "=====")
    # Fetch historical stock data
    stock_data = yf.download(company_symbol, start='2019-01-01', end='2023-09-20')

    # Feature Engineering
    stock_data['Price_Diff'] = stock_data['Close'].diff()
    stock_data['Price_Up'] = (stock_data['Price_Diff'] > 0).astype(int)

    # Drop NaN values and unnecessary columns
    stock_data.dropna(inplace=True)
    stock_data.drop(['Price_Diff', 'Adj Close', 'Volume'], axis=1, inplace=True)

    # Define features and target variable
    X = stock_data.drop('Price_Up', axis=1)
    y = stock_data['Price_Up']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the Support Vector Machine (SVM) model
    model = SVC(kernel='linear', random_state=42)
    model.fit(X_train, y_train)

    # Plot ROC Curve and Precision-Recall Curve
    plot_roc_and_pr_curves(model, X_test, y_test)

#### Function to plot Feature Importance Plot
def plot_feature_importance(X, model):
    result = permutation_importance(model, X, y, n_repeats=10, random_state=42)
    sorted_idx = result.importances_mean.argsort()
    plt.figure(figsize=(10, 6))
    plt.barh(X.columns[sorted_idx], result.importances_mean[sorted_idx])
    plt.xlabel("Permutation Importance")
    plt.title("Feature Importance")
    plt.show()

#### Loop through top company symbols and plot Feature Importance Plot
for company_symbol in top_companies:
    print("=====", company_symbol, "=====")
    # Fetch historical stock data
    stock_data = yf.download(company_symbol, start='2019-01-01', end='2023-09-20')

    # Feature Engineering
    stock_data['Price_Diff'] = stock_data['Close'].diff()
    stock_data['Price_Up'] = (stock_data['Price_Diff'] > 0).astype(int)

    # Drop NaN values and unnecessary columns
    stock_data.dropna(inplace=True)
    stock_data.drop(['Price_Diff', 'Adj Close', 'Volume'], axis=1, inplace=True)

    # Define features and target variable
    X = stock_data.drop('Price_Up', axis=1)
    y = stock_data['Price_Up']

    # Train the Support Vector Machine (SVM) model
    model = SVC(kernel='linear', random_state=42)
    model.fit(X, y)

    # Plot Feature Importance Plot
    plot_feature_importance(X, model)

    Accuracy for AAPL: 0.8319327731092437
    Classification Report for AAPL:
                  precision    recall  f1-score   support
    
               0       0.83      0.81      0.82       113
               1       0.83      0.86      0.84       125
    
        accuracy                           0.83       238
       macro avg       0.83      0.83      0.83       238
    weighted avg       0.83      0.83      0.83       238

    

    Accuracy for NKE: 0.7941176470588235
    Classification Report for NKE:
                  precision    recall  f1-score   support
    
               0       0.83      0.73      0.78       118
               1       0.76      0.86      0.81       120
    
        accuracy                           0.79       238
       macro avg       0.80      0.79      0.79       238
    weighted avg       0.80      0.79      0.79       238
    
    

    Accuracy for COST: 0.8361344537815126
    Classification Report for COST:
                  precision    recall  f1-score   support
    
               0       0.84      0.77      0.80       104
               1       0.83      0.89      0.86       134
    
        accuracy                           0.84       238
       macro avg       0.84      0.83      0.83       238
    weighted avg       0.84      0.84      0.84       238
    
    

    

    Accuracy for AMZN: 0.8109243697478992
    Classification Report for AMZN:
                  precision    recall  f1-score   support
    
               0       0.83      0.77      0.80       116
               1       0.79      0.85      0.82       122
    
        accuracy                           0.81       238
       macro avg       0.81      0.81      0.81       238
    weighted avg       0.81      0.81      0.81       238

#


## Linear Regression 

* performing the linear regression on only selected top most companies

```
# Filter the DataFrame for the specific brands
filtered_df = df.filter(df['Brand_Name'].isin('nike', 'amazon', 'apple', 'costco', 'honda'))

# Show some of the filtered data to verify
filtered_df.show()
display(filtered_df)
```
#### MOdel was trained for 3 months 



























## Web Scraping technique Yahoo Finance Most Active Stocks

This Python script scrapes data from Yahoo Finance's "Most Active" stocks page. It uses the `requests` library to fetch the HTML content of the webpage and the `BeautifulSoup` library to parse the HTML and extract relevant information.

### Scraping Function

The `scrape_yahoo_finance` function is defined to scrape data from a given URL. It sends an HTTP request with a specified user agent to mimic a browser and then parses the HTML response using BeautifulSoup.

### Initial URL

The script starts with the initial URL of Yahoo Finance's "Most Active" page.

### Scraping Loop

It enters a while loop to scrape data from multiple pages until there are no more "Next" buttons to navigate. Inside the loop, it:

*  Scrapes data from the current page using the `scrape_yahoo_finance` function.
* Finds all rows in the table and extracts data from each row, excluding the header row.
* Checks for the presence of a "Next" button to determine whether to continue scraping the next page.

## Data Storage

All scraped data is stored in a list named `all_data`.

## DataFrame Creation

After scraping all pages, the script creates a DataFrame from the collected data using Pandas. Column names for the DataFrame are predefined. The script drops the '52 Week Range' column as it's not needed.

## CSV Export

The DataFrame is saved to a CSV file named `Final_Web_Scrapping.csv` without an index column.

# Filtering Stocks

After scraping the data, the script filters the stocks based on a list of tickers obtained from another CSV file.

## Reading Tickers

It reads the CSV file containing the list of tickers.

## Filtering DataFrame

It filters the DataFrame to include only the rows where the ticker matches those in the ticker list.

## CSV Export

The filtered DataFrame is saved to a CSV file named `Final_Filtered_Web_Scrapping.csv` without an index column.

## Display

The filtered DataFrame is printed to the console.

```
# Import necessary libraries
import requests
from bs4 import BeautifulSoup
import pandas as pd
```

#### Function to scrape data from Yahoo Finance
```
def scrape_yahoo_finance(url):
    # Define headers for user agent to mimic a browser
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
    }
    # Send HTTP request to the URL and parse the HTML response
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.text, 'html.parser')
    return soup
```

#### Initial URL for Yahoo Finance's most active page
```
url = 'https://finance.yahoo.com/most-active?guce_referrer=aHR0cHM6Ly93d3cucGVycGxleGl0eS5haS8&guce_referrer_sig=AQAAAEpUE8Ie2uxufblRgzfbnoou7bhme0gn3VyN2h5YxM5VIp8UdIT_U0lYnrjYeB5AYsOVEpBaqwrkeaenS375N4rl9SpaGMCF1wI6QZG9NvsbO-A7csoTeNC2pfs7TNvhr28HmDmqBaBRwR6FpDGSMaQB1qQQ_h_rMiEsCsXlJ2Tc&offset=0&count=100'
```

##### List to store all scraped data
all_data = []

# Loop to scrape data from multiple pages
```
while True:
    # Scrape data from the current page
    soup = scrape_yahoo_finance(url)
    
    # Find all rows in the table
    rows = soup.find_all('tr')
    
    # Extract data from each row and append to all_data list
    for row in rows[1:]:  # Skip header row
        row_data = [data.text for data in row.find_all('td')]
        all_data.append(row_data)
    
    # Check for the presence of a "Next" button
    next_button = soup.find('a', class_='next')
    if next_button:
        # If "Next" button is found, update the URL and continue to the next page
        url = 'https://finance.yahoo.com' + next_button['href']
    else:
        # If "Next" button is not found, break out of the loop
        break
```

#### Define column names for the DataFrame
```
column_names = ['Symbol', 'Name', 'Price (Intraday)', 'Change', '% Change', 'Volume', 'Avg Vol (3 month)', 'Market Cap', 'PE Ratio (TTM)', '52 Week Range']

# Create DataFrame from the scraped data
df = pd.DataFrame(all_data, columns=column_names)

# Drop the '52 Week Range' column as it's not needed
df.drop(columns=['52 Week Range'], inplace=True)

# Save the DataFrame to a CSV file
df.to_csv('Final_Web_Scrapping.csv', index=False)

# Display the DataFrame
print(df)
```

#### Read the CSV containing the list of tickers
ticker_df = pd.read_csv("C:/Users/Varun/Desktop/AIT-614/Final_Project/World_Stock_Prices_1.csv")

#### Extract the tickers from the ticker_df
tickers = ticker_df['Ticker'].tolist()

#### Filter the DataFrame to include only the rows where the ticker matches those in your list
filtered_df = df[df['Symbol'].isin(tickers)]

#### a New file will be created with the necessary information
# Save the filtered DataFrame to a CSV file
filtered_df.to_csv('Final_Filtered_Web_Scrapping.csv', index=False)

#### Print the filtered DataFrame
print(filtered_df)

* After getting the file we can load the this dataset into our database and we can repeat the machine learning the steps to predict the stock market prices.


  


