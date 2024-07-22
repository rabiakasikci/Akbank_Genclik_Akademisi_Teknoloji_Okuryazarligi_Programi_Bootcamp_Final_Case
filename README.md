#Stock Prediction and Analysis Application
Overview
This application provides functionalities for fetching stock data, visualizing historical stock information, and predicting future stock prices using various models. It supports stocks listed on the BIST 30 index and includes the following features:

Viewing historical stock data
Plotting stock data (Open, High, Low, Adj Close, Close, Volume)
Performing stock price prediction using Linear Regression, ARIMA, LSTM, and Holt-Winters models
Displaying results and predictions in a Tkinter GUI
# Features
1) BIST30 Stock List
Displays a list of BIST30 stocks with their symbols and company names.

2) Stock Data Fetching
Fetches historical stock data from Yahoo Finance for a given stock symbol.

3) Stock Information Visualization
Plots historical stock data including Open, High, Low, Adj Close, and Close values.
Visualizes volume data in bar charts.
4) Stock Prediction
Linear Regression: Predicts stock prices using a linear regression model.
ARIMA: Uses the ARIMA model for time series forecasting.
LSTM: Employs Long Short-Term Memory (LSTM) neural networks for prediction.
ExponentialSmooting: Forecasts future stock prices using exponential smoothing.

5)  Future Prediction
Provides a comparison between predicted and actual values for future dates using ETS and LSTM models.

#Installation
To set up the environment, follow these steps:

Clone the repository:

git clone <repository-url>
cd <repository-directory>

Create and activate a virtual environment (optional but recommended):
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
Install the required packages:

pip install -r requirements.txt

#Requirements
Python 3.6 or later
Libraries:
numpy
pandas
matplotlib
seaborn
plotly
scipy
sklearn
statsmodels
pmdarima
keras
tensorflow
yfinance
tkinter
Pillow

#Usage
Run the application:
python Stock_Prediction_With_MachineLearning.py
Interact with the GUI:

Enter Stock Symbol: Input a stock symbol from the BIST 30 list.
Fetch Stock Data: Click on the button to retrieve and visualize historical stock data.
Predict Stock Prices: Perform predictions using different models and view results.
View Results:

Historical stock plots and predictions will be displayed in the Tkinter GUI.
Generated plots will be saved in the working directory.

#Code Structure
bist30_info(): Displays information about BIST 30 stocks.
fetch_stock_data(symbol): Fetches historical data for a given stock symbol.
stock_info(): Visualizes historical stock data.
stock_prediction(): Performs and visualizes stock price predictions using multiple models.
Contribution
Contributions are welcome! Please open an issue or submit a pull request if you have suggestions or improvements.

#License
This project is licensed under the MIT License. See the LICENSE file for details.

#Contact
For questions or feedback, please contact Rabia Kaşıkcı at rabiakasikci3@gmail.com.