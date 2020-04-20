
import pandas as pd
import numpy as np
import math, time, datetime

from alpha_vantage.timeseries import TimeSeries
from sklearn import preprocessing, svm
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from config import *


date = datetime.date.today().isocalendar()[:-1]

weekly_tickers = f"C:\\Users\\lm44\\Documents\\Code\\Python\\Trading\\Data\\ml-vwap-ticker{date}.csv"
linear_regression_tickers = "C:\\Users\\lm44\\Documents\\Code\\Python\\Trading\\Data\\Linear Regression Tickers.csv"

time_series_data = TimeSeries(alphavantage_key, output_format='pandas')


def get_historic_data():

    ticker_df = pd.read_csv(weekly_tickers)
    ticker_list = [ticker for ticker in ticker_df['Symbol'].values]
    tickers = []

    for ticker in ticker_list:
        count = ticker_list.count(ticker)
        if count > 2:
            tickers.append(ticker)
        
    time_series_data = TimeSeries(alphavantage_key, output_format='pandas') 

    dict_of_dataframes = {}
    print(tickers)
    for ticker in set(tickers):

        df = time_series_data.get_daily(symbol=ticker, outputsize='full')[0].iloc[::-1].reset_index()
        df = df[['1. open', '2. high', '3. low', '4. close', '5. volume']]
        dict_of_dataframes.update({ticker:df})
        time.sleep(12.00001)

    return(dict_of_dataframes)

print(get_historic_data())
def linear_regression():
    data = pd.DataFrame(columns=['tickers'])

    for ticker, df in get_historic_data().items():
        
        df['High - Low'] = df['2. high'] - df['3. low']
        df['Percent Change'] = ((df['4. close'] - df['1. open']) / df['1. open']) * 100
        df = df[['4. close', '5. volume', 'Percent Change', 'High - Low', 's&p change']]
        df.dropna(inplace=True) 
        df = df.copy()

        forecast = '4. close'
        forecast_length = int(math.ceil( (len(df)/len(df)*5) ))
        df['label'] = df[forecast].shift(-forecast_length)

        X = np.array(df.drop(['label'], 1))
        X = preprocessing.scale(X)
        X_forecasted = X[-forecast_length:]
        X = X[:-forecast_length]
        df.dropna(inplace=True)
        df = df.copy()
        y = np.array(df['label'])

        X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0)

        clf = LinearRegression()
        clf.fit(X_train, y_train)

        score = clf.score(X_test, y_test)
        forecasted_values = clf.predict(X_forecasted)
        
        if (score > 0.9 and score < 1) and (forecasted_values[-1] > df['4. close'].iloc[-1]):
            print(f"{ticker}: forecast {forecasted_values}")
            data = data.append({'tickers': ticker}, ignore_index=True)


    data.to_csv(linear_regression_tickers)    

# linear_regression()