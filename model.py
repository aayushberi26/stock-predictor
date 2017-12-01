import pandas as pandas
from pandas_datareader import data as web
import datetime
import numpy as np
'''
 [simple_model_vectors (d, ticker, startDate)] takes a stock ticker, number of days to evaluate, and the start_date and generates
 a feature vector (list) of the following form
 [
	open_day_1,
	high_day_1,
	low_day_1,
	adj_close_day_1,
	volume_day_1,
		.
		.
		.
	volume_day_n
 ]
'''
def simple_model_vectors (ticker, d, startDate):
	start = startDate
	end = startDate + datetime.timedelta(days=d-1)
	df = web.DataReader(ticker, 'yahoo', start, end)
	df.drop(['Close'], 1, inplace=True)
	return list(df.values.flatten())

'''
 [advanced_model_vectors (d, ticker, startDate)] takes a stock ticker, number of days to evaluate, and the start_date and generates
 a feature vector (list) of the following form
 [
	open_day1,
	return_day_1,
	volume_diff_day_1,
	volatility_day_1
		.
		.
		.
	volatility_day_n
 ]
'''
def advanced_model_vectors (ticker, d, startDated):
	start = startDate - datetime.timedelta(days=30)
	end = startDate+datetime.timedelta(days=d-1)
	df = web.DataReader(ticker, 'yahoo', start, end)
	df.drop(['Adj Close', 'High', 'Low'], 1, inplace = True)
	df['Return'] = (df['Close'] - df['Open'])/df['Open']
	df.drop(['Close'], 1, inplace=True)
	df['Volume Diff'] = df.diff()['Volume']
	df.drop(['Volume'], 1, inplace = True)
	df['Moving Average'] = df.rolling(window=10).mean()['Open']
	df['Deviation'] = df['Open'] - df['Moving Average']
	df['Deviation Sq'] = df['Deviation'] * df['Deviation']
	df['Volatility Sq'] = df.rolling(window=10).mean()['Deviation Sq']
	df['Volatility'] = df['Volatility Sq'].apply(np.sqrt)
	df.drop(['Moving Average', 'Deviation', 'Deviation Sq', 'Volatility Sq'], 1, inplace = True)
	df = df.ix[-1*d:]
	return list(df.values.flatten())