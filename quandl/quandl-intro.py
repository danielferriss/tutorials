import quandl
import pandas as pd

quandl.ApiConfig.api_key = 'NdyospgCKrp_eo-jRuU7'

data = quandl.get('NSE/OIL')

print(data)