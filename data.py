import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt

sensex = pd.read_csv("Sensex.csv", on_bad_lines='skip')
sensex.columns = ['Date', 'Price', 'Open', 'High', 'Low', 'Vol.', 'Change %']
gold = pd.read_csv("Gold.csv", on_bad_lines='skip')
oil = pd.read_csv("Crude_Oil.csv", on_bad_lines='skip')

def stock(df, date_col='Date', price_col='Close'):
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce',format='mixed')
    df = df.dropna(subset=[date_col])
    df = df.set_index(date_col)
    df = df.sort_index()  
    if price_col not in df.columns:
        raise ValueError(f"Column '{price_col}' not found. Columns available: {df.columns.tolist()}")
    df[price_col] = df[price_col].astype(str).str.replace(',', '').astype(float)
    return df[price_col]

sensex = stock(sensex, price_col="Price")
gold = stock(gold, price_col="Price")
oil = stock(oil, price_col="Price")

def add_ma(series):
    df = pd.DataFrame(series)
    df['MA20'] = series.rolling(20).mean()
    df['MA50'] = series.rolling(50).mean()
    return df

sensex_ma = add_ma(sensex)
gold_ma = add_ma(gold)
oil_ma = add_ma(oil)

def plotma(df, name):
    plt.figure(figsize=(12,5))
    plt.plot(df.index, df.iloc[:,0], label='Price')
    plt.plot(df.index, df['MA20'], label='MA20')
    plt.plot(df.index, df['MA50'], label='MA50')
    plt.title(name)
    plt.legend()
    plt.show()

plotma(sensex_ma, "Sensex (MA)")
plotma(gold_ma, "Gold (MA)")
plotma(oil_ma, "Crude Oil (MA)")

def runarima(series, name, order=(5,1,2), forecast_days=50):
    model = ARIMA(series, order=order)
    res = model.fit()
    
    forecast = res.get_forecast(steps=forecast_days)
    forecast_df = forecast.summary_frame()
    last_date = series.index[-1]
    future_index = pd.date_range(start=last_date, periods=forecast_days+1, freq='D')[1:]
    forecast_df.index = future_index
    
    plt.figure(figsize=(10,4))
    plt.plot(series, label='Actual')
    plt.plot(forecast_df['mean'], label='Forecast')
    plt.title(f"{name} Forecast")
    plt.legend()
    plt.show()


sensexforecast = runarima(sensex, "Sensex")
goldforecast = runarima(gold, "Gold")
oilforecast = runarima(oil, "Crude Oil")
