from alpha_vantage.timeseries import TimeSeries
from config import alphavantage_key
from sklearn import preprocessing, svm
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import time, math

yahoo_url = "https://uk.finance.yahoo.com/screener/predefined/day_gainers"


def get_universe():
    #call daily
    df = pd.read_html(yahoo_url)[0]
    df['Symbol'].to_csv("C:\\Users\\lm44\\Documents\\Code\\Python\\Trading\\Data\\ml-vwap-ticker.csv")

    


def get_historic_data():
    #call weekly before machine learning

    ticker_df = pd.read_csv("C:\\Users\\lm44\\Documents\\Code\\Python\\Trading\\Data\\ml-vwap-ticker.csv")
    ticker_list = list(ticker_df) #take the stocks that appear on yahoo the most - list.count()
    print(ticker_list)
    time_series_data = TimeSeries(alphavantage_key, output_format='pandas')   
    dict_of_dataframes = {}

    for ticker in ticker_list[:3]:
        data = time_series_data.get_daily(symbol=ticker, outputsize='full')
        data = data[0].iloc[::-1].reset_index(drop=True)
        dict_of_dataframes.update({ticker:data})
        time.sleep(12.00001)

    return(dict_of_dataframes)


def linear_regression():
    results = {}

    for ticker, df in get_historic_data().items():
        
        df['Percent Change'] = ((df['4. close'] - df['1. open']) / df['1. open']) * 100
        df['High - Low'] = df['2. high'] - df['3. low']
        df = df[['4. close', '5. volume', 'Percent Change', 'High - Low']]
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
        
        if score > 0.95 and score < 1:
            print(forecasted_values)
            results.update({ticker: df})


    return(results)

print(linear_regression())