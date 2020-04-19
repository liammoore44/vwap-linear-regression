from alpha_vantage.timeseries import TimeSeries
from alpha_vantage.techindicators import TechIndicators
from config import *
from sklearn import preprocessing, svm
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from twilio.rest import Client
from itertools import product

import pandas as pd
import numpy as np
import time, datetime, math, requests, json, multiprocessing, csv


date = datetime.date.today().isocalendar()[:-1]

yahoo_url = "https://uk.finance.yahoo.com/screener/predefined/day_gainers"

headers = {"APCA-API-KEY-ID": alpaca_key, "APCA-API-SECRET-KEY": secret_key}
client = Client(twilio_sid, twilio_auth)

indicators = TechIndicators(alphavantage_key, output_format='pandas')
time_series_data = TimeSeries(alphavantage_key, output_format='pandas')
ticker = 'KGC'


def get_universe():
    #call daily
    df = pd.read_html(yahoo_url)[0]
    old_df = pd.read_csv(f"C:\\Users\\lm44\\Documents\\Code\\Python\\Trading\\Data\\ml-vwap-ticker\\{date}.csv")
    df = old_df.append(df)
    df.to_csv(f"C:\\Users\\lm44\\Documents\\Code\\Python\\Trading\\Data\\ml-vwap-ticker{date}.csv")


def get_historic_data():

    ticker_df = pd.read_csv(f"C:\\Users\\lm44\\Documents\\Code\\Python\\Trading\\Data\\ml-vwap-ticker{date}.csv")
    ticker_list = [ticker for ticker in ticker_df['Symbol'].values]
    tickers = []

    for ticker in ticker_list:
        count = ticker_list.count(ticker)
        if count > 1:
            tickers.append(ticker)
        
    time_series_data = TimeSeries(alphavantage_key, output_format='pandas') 

    dict_of_dataframes = {}

    for ticker in tickers[:6]:
        data = time_series_data.get_daily(symbol=ticker, outputsize='full')
        data = data[0].iloc[::-1].reset_index(drop=True)
        dict_of_dataframes.update({ticker:data})
        time.sleep(12.00001)

    return(dict_of_dataframes)


def linear_regression():
    data = pd.DataFrame(columns=['tickers'])

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
        
        if score > 0.8 and score < 1:
            print(f"{ticker}: forecast {forecasted_values}")
            data = data.append({'tickers': ticker}, ignore_index=True)


    data.to_csv("C:\\Users\\lm44\\Documents\\Code\\Python\\Trading\\Data\\Linear Regression Tickers.csv")    


def get_clock():
    request = requests.get(clock_url, headers=headers)
    return json.loads(request.content)


def get_stock_data(ticker):
    parameters = {
    "apikey": td_key,
    "symbol": ticker
    }
    request = requests.get(url=td_quotes_url, params=parameters).json()
    data = pd.DataFrame.from_dict(request, orient='index').reset_index(drop=True)
    
    return data['lastPrice']


def make_df(ticker):
    
    vwap = indicators.get_vwap(symbol=ticker, interval='15min',)[0]
    data = time_series_data.get_intraday(symbol=ticker, interval='15min', outputsize='full')
    data = data[0].iloc[::-1]
    data[f'{ticker}pct_change'] = (data['4. close'] - data['1. open'])/ data['1. open'] * 100
    df = data.join(vwap)
    df[f'{ticker} price/vwap range'] = df['4. close'] - df['VWAP']
    df.to_csv(f"C:\\Users\\lm44\\Documents\\Code\\Python\\Trading\\Data\\{ticker}.csv", index=True)
    global ranges
    ranges = [value for value in df[f'{ticker} price/vwap range'].values[-220:]]

    return(df[f'{ticker} price/vwap range'][-1], df['4. close'][-1])


def check_vwap(ticker):

    while get_clock()["is_open"] == False:

        new_data = make_df(ticker)
        new_range = new_data[0]
        largest_range = max(ranges)

        if new_range == largest_range:
            print('watch function triggered')
            trigger_buy()
            profit_loss()
        else:
            print('...')
            time.sleep(900.1)


def trigger_buy():

    while get_clock()["is_open"] == True:

        new_data = make_df(ticker)

        if (new_data[1] <=  new_data[0]*1.03) and [range > 0 for range in ranges[-3:]]:
            client.messages.create(
                to = '+4407860209951',
                from_= '+13343453192',
                body=f'\nBUY SIGNAL FOR {ticker} TRIGGERED AT ${new_data[1]}.'
            )
            print('buy function triggered')
            # row = owned_stocks_df.loc[owned_stocks_df['Stock'] == ticker]
            # row['Owned'] = 'True'
            global entry_price 
            entry_price = new_data[1]
            break
        else:
            print('-')
            time.sleep(900.1)


def profit_loss():

    while get_clock()['is_open'] == True:
        
        last_price = get_stock_data(ticker)

        if last_price <= entry_price*0.94:
            client.messages.create(
                to = '+4407860209951',
                from_= '+13343453192',
                body=f'\nSTOP LOSS FOR {ticker} TRIGGERED AT ${last_price}.'
            )
            # row = owned_stocks_df.loc[owned_stocks_df['Stock'] == ticker]
            # row['Owned'] = 'False'
        elif last_price >= entry_price*1.1:
            client.messages.create(
                to = '+4407860209951',
                from_= '+13343453192',
                body=f'\nPOSITION IN {ticker} AT 10% PROFIT. - PRICE({last_price})'
            )
        elif last_price >= entry_price*1.15:
            client.messages.create(
                to = '+4407860209951',
                from_= '+13343453192',
                body=f'\nSTOP LOSS TRIGGERED FOR {ticker} AT 15% PROFIT. - PRICE({last_price})'
            )        
            # row = owned_stocks_df.loc[owned_stocks_df['Stock'] == ticker]
            # row['Owned'] = 'False'
        else:
            time.sleep(20)

dataframe = pd.read_csv("C:\\Users\\lm44\\Documents\\Code\\Python\\Trading\\Data\\Linear Regression Tickers.csv")
use_tickers = list(dataframe['tickers'])
    
if __name__ == '__main__':

    get_universe()
 
    with multiprocessing.Pool() as pool:
        results = pool.starmap(check_vwap, product(use_tickers[:5]))
        print(results)

