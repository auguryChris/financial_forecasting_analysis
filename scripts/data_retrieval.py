#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#%%
# Import libraries
import requests
import pandas as pd
import numpy as np
from datetime import date, datetime
from pandas.tseries.offsets import BDay
import sys, os
import pickle

#!pip install quandl
import quandl

#!pip install yfinance
import yfinance as yf

#!pip install pandas-datareader
from pandas_datareader import data as pdr

# API Credentials
quandl.ApiConfig.api_key = 'ufyMTAbgF8LmdWSFWRDT'

absolutepath = os.path.abspath(__file__)
fileDirectory = os.path.dirname(absolutepath)

#Path of parent directory
parentDirectory = os.path.dirname(fileDirectory)

# S&P 500 Scraper
def sp500_list_retrieval():
    """
    Retrieve the S&P 500 list from the corresponding Wikipedia page.

    Returns a dataframe of all tickers and corresponding industry information.
    """
    print("Acquiring S&P500 Tickers list.")

    table = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')

    df = table[0]
    df.columns = ['Symbol',
                  'Security',
                  'SEC filings',
                  'GICS Sector',
                  'GICS Sub Industry',
                  'Headquarters Location',
                  'Date First Added',
                  'CIK',
                  'Founded'
                   ]
    df.drop(labels=['SEC filings',
                    'Headquarters Location',
                    'Date First Added',
                    'CIK'
                    ],
              axis=1,
              inplace=True
              )

    df.to_csv(f"{parentDirectory}/data/SP500_list.csv")
    return df

# Fundamental Data Pull from Quandl
def fundamentals_data_pull(timeframe='MRT', ticker=[], start_yr='2016', end_date=None):
    """Pull the fundamentals data for each ticker passed through. If no timeframe
    is selected, all timeframes (dimensions) will be pulled. Otherwise, timeframes
    may be selected from one of the following:

    Select One:
    - AR (As Reported)
    - MR (Most-Recent Reported)

    + One of the Following
    - Y (Annual)
    - T (Trailing Twelve Months)
    - Q (Quarterly)
    (i.e. MRY = Most-Recent Reported Annual Data)

    Keyword Arguments:
    timeframe -- one of the above (default 'MRT')
    ticker -- list of stock tickers (default empty list)

    Returns:
    dict -- keys are tickers, values are dataframe of all Quandl fundamentals data for designated timeframe
    """

    print("Acquiring fundamentals data for tickers.")

    total_length = 0

    # Reduce dataset only to years requested
    cutoff_date = str(start_yr + "-01-01")
    #cutoff_date = pd.to_datetime(start_yr + "-01-01")
    #df = df[df['reportperiod'] > cutoff_date]

    if timeframe == None:
        timeframe = input("Please select a timeframe from 'MRT', 'ART','ARY', or 'MRY':  ")
        #df = quandl.get_table('SHARADAR/SF1', ticker=ticker)
        # Removing to eliminate confusion with Quarterly results and potential incompatibility down the line -- AB 9/14

    elif timeframe in ['MRT', 'ART','ARY', 'MRY']:
        if end_date == None:
            end_date = str(date.today())
            df = quandl.get_table('SHARADAR/SF1', dimension=timeframe, calendardate={'gte': cutoff_date, 'lte' : end_date}, ticker=ticker, paginate=True)
        else:
            df = quandl.get_table('SHARADAR/SF1', dimension=timeframe, calendardate={'gte': cutoff_date, 'lte' : end_date}, ticker=ticker, paginate=True)

    elif timeframe in ['MRQ', 'ARQ']:
        raise ValueError("Quarterly data is not compatible with this analysis. Please select a timeframe from 'MRT', 'ART','ARY', or 'MRY'")

    else:
        timeframe = input("Please select a timeframe from 'MRT', 'ART','ARY', or 'MRY':  ")


    # Create a dictionary where keys are tickers and values are dataframes of fundamentals
    fund_dict = {}
    for x in ticker:
        df0 = df.copy()
        df0 = df0[df0['ticker'] == x.upper()]
        if len(df0) == 0:
            print("No data provided for symbol '" + x.upper() + "'")
            pass
        else:
            fund_dict[x.upper()] = df0
            total_length += len(df0)

    print("total length of fundamentals data dict: " + str(len(fund_dict.keys())))
    print("total length of fundamentals data: " + str(total_length))
    return fund_dict


# Pricing Data Pull from yfinance
def ohlcv_data(fund_dict):
    """Pull the pricing data for each ticker.

    Keyword Arguments:
    fund_dict -- dict, dictionary where k,v pairs are ticker,fundamental_data pairs

    Returns:
    dict -- keys are tickers
    """

    class HiddenPrints():
        def __enter__(self):
            self._original_stdout = sys.stdout
            sys.stdout = open(os.devnull, 'w')

        def __exit__(self, exc_type, exc_val, exc_tb):
            sys.stdout.close()
            sys.stdout = self._original_stdout

    print("Acquiring pricing data for tickers.")

    total_length = 0

    #current_price = {}
    to_del = []

    # Set business days
    isBusinessDay = BDay().is_on_offset

    # Iterate through dictionary pulling data for that symbol, joining to
    # corresponding fundamentals dataframe.
    for k in fund_dict:
        start_date = min(fund_dict[k]['reportperiod'])
        #end_date = max(fund_dict[k]['reportperiod'])
        end_date = date.today()

        with HiddenPrints():
            yf.pdr_override()
            data = pdr.get_data_yahoo(k, start=start_date, end=end_date)
#            data = yf.download(k, start=start_date, end=end_date)

        if len(data) > 0:
        #    current_price = data['Close'].iloc[-1]
            data.reset_index(inplace=True)
            data.rename(columns={"Date" : "reportperiod"}, inplace=True)

            # Sometimes quarter end/reporting period end is not on trading day.
            # To do this, we will do an outer join of all data, sort by date, backfill data, then eliminate non-trading days
            fund_dict[k] = pd.merge(fund_dict[k], data, left_on='reportperiod', right_on='reportperiod', how='outer', sort=False)
            fund_dict[k].sort_values('reportperiod',ascending=False,inplace=True)
            fund_dict[k] = fund_dict[k].bfill()

            # Filter for just trading days
            match_series = pd.to_datetime(fund_dict[k]['reportperiod']).map(isBusinessDay)
            fund_dict[k] = fund_dict[k][match_series]

            total_length += len(fund_dict[k])

        elif len(data) == 0:
            to_del.append(k)
            pass

    print("Deleting tickers " + str(to_del) + " from dataset due to insufficient pricing data.")
    for k in to_del:
        del fund_dict[k]

    print("total length of fundamentals data dict: " + str(len(fund_dict.keys())))
    print("total length of fundamentals data: " + str(total_length))
    print("S&P Ticker Count after Data Acquisition:  " + str(len(fund_dict.keys())))

    return fund_dict


# Data Collector
def data_collection(timeframe='MRT', ticker=[], start_yr='2016', end_date=None):

    df = sp500_list_retrieval()
    stock_list = list(df['Symbol'].unique())
    print("S&P 500 Ticker Count:  " + str(len(stock_list)))

    # Retrieve fundamentals data for all S&P 500 stocks
    stock_dict = fundamentals_data_pull(timeframe=timeframe, ticker=stock_list, start_yr=start_yr, end_date=None)

    # Merge pricing data for all stocks
    stock_dict = ohlcv_data(stock_dict)

    # Final data frame
    final = pd.concat(stock_dict.values(), ignore_index=True)
    final = final.sort_values(by=['reportperiod','ticker'])

    return final

#x = data_collection()

if __name__ == '__main__':
#    import argparse
#
#    parser = argparse.ArgumentParser
#    parser.add_argument('output_file', help='raw dataset')
#
#    args = parser.parse_args()
#
    # Collect data
    df = data_collection(start_yr='2000') 
    
    # get date stamp
    dateTimeObj = datetime.now().strftime("%Y-%m-%d_%H.%M.%S")

    with open(f'{parentDirectory}/data/raw_data_{dateTimeObj}.pkl', 'wb+') as out:
        pickle.dump(df, out)

# %%
