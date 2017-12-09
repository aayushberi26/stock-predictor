import pandas as pandas
from pandas_datareader import data as web
import datetime
import numpy as np
from random import *
import csv
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
    end = startDate + datetime.timedelta(days=30)
    df = web.DataReader(ticker, 'yahoo', start, end)
    df.drop(['Adj Close'], 1, inplace=True)
    endOpen = df.ix[d].Open
    endClose = df.ix[d].Close
    df = df.ix[:d]
    lst = list(df.values.flatten())
    if endClose > endOpen:
        lst.append(1.0)
        return lst
    else:
        lst.append(0.0)
        return lst

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
def advanced_model_vectors (ticker, d, startDate):
    start = startDate - datetime.timedelta(days=30)
    end = startDate+datetime.timedelta(days=30)
    df = web.DataReader(ticker, 'yahoo', start, end)
    df.drop(['Adj Close', 'High', 'Low'], 1, inplace = True)
    df['Return'] = (df['Close'] - df['Open'])/df['Open']
    df['Volume Diff'] = df.diff()['Volume']
    df.drop(['Volume'], 1, inplace = True)
    df['Moving Average'] = df.rolling(window=10).mean()['Open']
    df['Deviation'] = df['Open'] - df['Moving Average']
    df['Deviation Sq'] = df['Deviation'] * df['Deviation']
    df['Volatility Sq'] = df.rolling(window=10).mean()['Deviation Sq']
    df['Volatility'] = df['Volatility Sq'].apply(np.sqrt)
    df.drop(['Moving Average', 'Deviation', 'Deviation Sq', 'Volatility Sq'], 1, inplace = True)
    df = df.loc[startDate:,:]
    endOpen = df.ix[d].Open
    endClose = df.ix[d].Close
    df = df.ix[:d]
    df.drop(['Close'], 1, inplace=True)
    if endClose > endOpen:
        return list(df.values.flatten()).append(1)
    else:
        return list(df.values.flatten()).append(0)

'''
[random_date ()] generates a random date between 6/1/2007 and 6/1/2017. 
This represents a 10 year period of dates
'''
def random_date ():
    start = datetime.datetime(2007,6,1)
    randDays = randint(0,365*10)
    return (start+datetime.timedelta(days=randDays))

def output_simple (ticker):
    file = open(ticker+'Simple.csv', 'w')
    allData = []
    i = 0
    while i < 5000:
        try:
            allData.append(simple_model_vectors(ticker, 10, random_date()))
            print("Vector: " + str(i+1) + "/5000")
            i = i+1
        except:
            continue
    with file:
        writer = csv.writer(file)
        writer.writerows(allData)

def import_data(file):
    with open(file) as myFile:
        reader = csv.reader(myFile)
        allData = []
        for row in reader:
            allData.append(row)
        return allData

def output_advanced (ticker):
    ticker = ticker.lower()
    file = open(ticker+'Simple.csv', 'w')
    allData = []
    i = 0
    while i < 5000:
        try:
            allData.append(advanced_model_vectors(ticker, 10, random_date()))
            print("Vector: " + str(i+1) + "/5000")
            i = i+1
        except:
            continue
    with file:
        writer = csv.writer(file)
        writer.writerows(allData)