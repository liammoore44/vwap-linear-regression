from alpha_vantage.timeseries import TimeSeries
from config import alphavantage_key
import pandas as pd
import time

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

print(get_historic_data())
