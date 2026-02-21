
from arch import arch_model
from arch.__future__ import reindexing
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# IMPORT your modules
from processing.data_cleaner import clean_data, create_features

# IMPORT MT5 download function
from ingestion.mt5_download import download_data
from ingestion.mt5_download import download_data

symbols = ["EURUSD", "GBPUSD", "NZDUSD"]

results = {}

for symbol in symbols:

    df = download_data(symbol, "H4")


df['Return'] = 100 * (df['Close'].pct_change())

#____________ Volatility__________________________
daily_volatility = data['Return'].std()
print('Daily volatility: ', '{:.2f}%'.format(daily_volatility))

monthly_volatility = math.sqrt(21) * daily_volatility
print ('Monthly volatility: ', '{:.2f}%'.format(monthly_volatility))

annual_volatility = math.sqrt(252) * daily_volatility
print ('Annual volatility: ', '{:.2f}%'.format(annual_volatility ))

#_____________GARCH model__________________________
garch_model = arch_model(df['Return'], p = 1, q = 1,
                      mean = 'constant', vol = 'GARCH', dist = 'normal')

gm_result = garch_model.fit(disp='off')
print(gm_result.params)

print('\n')

gm_forecast = gm_result.forecast(horizon = 5)
print(gm_forecast.variance[-1:])

# Plotting the volatility forecast
plt.figure(figsize=(10, 6)) 
plt.plot(gm_forecast.variance[-1:].T, marker='o')
plt.title('GARCH(1,1) Volatility Forecast')
plt.xlabel('Horizon (days)')
plt.ylabel('Forecasted Variance')

#_________________________ Plotting the volatility forecast_________________________
plt.figure(figsize=(12,4))
plt.plot(stock_data['Return'][-365:])
plt.plot(rolling_predictions)
plt.title('Volatility Prediction - Rolling Forecast')
plt.legend(['True Daily Returns', 'Predicted Volatility'])
plt.show()