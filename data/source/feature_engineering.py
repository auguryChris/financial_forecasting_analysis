#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  2 10:58:53 2021

@author: Allie
"""
#%%
#import pickle
import ta
import pandas as pd
import os

path = os.getcwd()
path = path + '/financial_forecasting_analysis/data/data/og_raw_data.pkl'

df = pd.read_pickle(path)
#df = df.head(100000)

def add_features(df):

    windows = [6,18,24,30,50,100,200]

    all_tickers = []

    for ticker, df0 in df.groupby('ticker'):
        df0 = df0.copy()

        for w in windows:
            if len(df0) >= w:
                # RSI
                df0['RSI_' + str(w)] = ta.momentum.RSIIndicator(df0['Close'], window=w).rsi()


                # MACD
                for w2 in windows:
                    if w > w2:
                        # Will utilize macd_diff because that is more normalized
                        df0['MACD_f'+str(w2)+'_s'+str(w)] = ta.trend.MACD(df0['Close'], window_slow=w2, window_fast=w).macd_diff()

                # Bollinger Bands
                ## Stdev default=2, but can change it if desired
                # Currently returning high/low band indicators, but can add actual values if desired.
                bbands = ta.volatility.BollingerBands(df0['Close'], window=w)
                df0['BBands_' + str(w) + '_h_ind'] = bbands.bollinger_hband_indicator()
                df0['BBands_' + str(w) + '_l_ind'] = bbands.bollinger_lband_indicator()

                # actual values
    #            df0['BBands_' + str(w) + '_h_ind'] = bbands.bollinger_hband()
    #            df0['BBands_' + str(w) + '_l_ind'] = bbands.bollinger_lband()

                # Average True Range (ATR)
                df0['ATR_' + str(w)] = ta.volatility.AverageTrueRange(high=df0['High'],low=df0['Low'],close=df0['Close'], window=w).average_true_range()

                # Stochastic Oscillator (SR/STOCH)
                df0['STOCH_' + str(w)] = ta.momentum.StochasticOscillator(high=df0['High'],low=df0['Low'],close=df0['Close'], window=w).stoch()

                # Chaikin Money Flow Indicator (CMF)
                df0['CMF_' + str(w)] = ta.volume.ChaikinMoneyFlowIndicator(high=df0['High'],low=df0['Low'],close=df0['Close'],volume=df0['Volume'], window=w).chaikin_money_flow()

                # Ichimoku Indicator (ICHI)
                for w2 in windows:
                    for w3 in windows:
                        if (w > w2) & (w2 > w3):
                            ichimoku = ta.trend.IchimokuIndicator(high=df0['High'],low=df0['Low'],window1=w3, window2=w2, window3=w)
                            df0['ICHI_conv_' + str(w3)+'_'+str(w2)+'_'+str(w)] = ichimoku.ichimoku_conversion_line()
                            df0['ICHI_base_' + str(w3)+'_'+str(w2)+'_'+str(w)] = ichimoku.ichimoku_base_line()
                            df0['ICHI_diff_' + str(w3)+'_'+str(w2)+'_'+str(w)] = df0['ICHI_conv_' + str(w3)+'_'+str(w2)+'_'+str(w)] - df0['ICHI_base_' + str(w3)+'_'+str(w2)+'_'+str(w)]


                # SMA
                df0['SMA_' + str(w)] = ta.trend.SMAIndicator(df0['Close'], window=w).sma_indicator()

                # SMA Crossover
                for w2 in windows:
                    if w > w2:
                        sma_s = ta.trend.SMAIndicator(df0['Close'], window=w).sma_indicator()
                        sma_f = ta.trend.SMAIndicator(df0['Close'], window=w2).sma_indicator()
                        df0['SMA_cross_f' + str(w2) + '_s' + str(w)] = sma_f - sma_s

                # EMA
                df0['EMA_' + str(w)] = ta.trend.EMAIndicator(df0['Close'], window=w).ema_indicator()

                # EMA Crossover
                for w2 in windows:
                    if w > w2:
                        ema_s = ta.trend.EMAIndicator(df0['Close'], window=w).ema_indicator()
                        ema_f = ta.trend.EMAIndicator(df0['Close'], window=w2).ema_indicator()
                        df0['SMA_cross_f' + str(w2) + '_s' + str(w)] = ema_f - ema_s


            ## WINDOW NOT REQUIRED
            # On Balance Volume Indicator (OBV)
            df0['OBV'] = ta.volume.OnBalanceVolumeIndicator(close=df0['Close'],volume=df0['Volume']).on_balance_volume()

            # Volume-Price Trend (VPT)
            df0['VPT'] = ta.volume.VolumePriceTrendIndicator(close=df0['Close'],volume=df0['Volume']).volume_price_trend()

            # Accumulation/Distribution Index Indicator (ADI)
            df0['ADI'] = ta.volume.AccDistIndexIndicator(high=df0['High'],low=df0['Low'],close=df0['Close'],volume=df0['Volume']).acc_dist_index()

        all_tickers.append(df0)

    final = pd.concat(all_tickers)
    final = final.sort_values(by=['reportperiod','ticker'])

    return final

df = add_features(df)