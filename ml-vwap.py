import pandas as pd

yahoo_url = "https://uk.finance.yahoo.com/screener/predefined/day_gainers"


def get_universe():
    #call daily
    df = pd.read_html(yahoo_url)[0]
    df['Symbol'].to_csv("C:\\Users\\lm44\\Documents\\Code\\Python\\Trading\\Data\\ml-vwap-ticker.csv")

    

