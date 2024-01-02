# -*- coding: utf-8 -*-
"""
Created on Tue Jan  2 09:46:11 2024

@author: USER
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# #%%
# def get_sentiment_indicator(param_set, price_df):
#     def get_rank(array):
#         # The higher, the lower rank
#         s = pd.Series(array)
#         return s.rank(ascending=False).iloc[len(s)-1]
    
#     lookback_ticker = param_set
        
#     ind_df = dict()
    
#     for lookback in lookback_ticker:
#         print(lookback)
#         rank = price_df.rolling(lookback).apply(get_rank)
#         rank['avg'] = rank.mean(axis=1) / lookback
#         rank.index = pd.to_datetime(rank.index)
#         ind_df[lookback] = rank['avg']
        
#     return ind_df

# sent_df = pd.read_feather('sentiment_prices_df.feather')
# sent_df.index = pd.to_datetime(sent_df.index)

# ind_df = get_sentiment_indicator([20, 240], sent_df)

#%%
# os.chdir('C:\\Users\\USER\\Desktop\\Python')
df = pd.read_feather('https://github.com/davidkim0523/QIS-Dashboard/blob/main/qis_df.feather')


strat_name = ['Equity Beta', 'Dynamic Put Ratio', 'Vol Roll on Rate', 'Bond Momentum',
              'FX Carry', 'Commodity Carry', 'Commodity Momentum', 'Commodity Value',
              'Equity Volatility']

df.columns = strat_name

rets = df.pct_change().dropna()
vols = rets.rolling(60).std() * np.sqrt(252)
iv_weights = (1/vols).div((1/vols).sum(axis=1), axis=0)
iv_weights_monthly = iv_weights.resample('BM').last()
iv_weights_monthly = iv_weights_monthly.reindex_like(iv_weights)
iv_weights_monthly = iv_weights_monthly.ffill()

strat_rets = iv_weights_monthly.shift() * rets
port_rets = strat_rets.sum(axis=1)

# plt.figure(figsize=(12, 7))
# plt.plot(port_rets.cumsum())
# plt.show()

#%% Cluster Backtesting
offensive = strat_rets[['Equity Beta', 'FX Carry', 'Equity Volatility']]
interest = strat_rets[['Bond Momentum', 'Vol Roll on Rate']]
defensive = strat_rets[['Commodity Value', 'Dynamic Put Ratio', 'Commodity Momentum', 'Commodity Carry']]

res_df = pd.concat([offensive.sum(axis=1), interest.sum(axis=1), defensive.sum(axis=1)], axis=1, join='inner')
res_df.columns = ['offensive', 'rates', 'defensive']

# # Portfolio
# res_df.loc['2023':].cumsum().plot(figsize=(12,7))
# res_df.loc['2023':].sum(axis=1).cumsum().plot(figsize=(12,7))
# plt.show()

# offensive.loc['2023':].cumsum().plot(figsize=(12,7))
# plt.show()

#%% Performance Table
ret_1d = rets.iloc[-1]
ret_1w = rets.iloc[-5:].sum()
ret_1m = rets.iloc[-20:].sum()
ret_3m = rets.iloc[-60:].sum()
ret_6m = rets.iloc[-120:].sum()
ret_ytd = rets.loc['2023':].sum()

perf_df = pd.concat([ret_1d, ret_1w, ret_1m, ret_3m, ret_6m, ret_ytd], axis=1)
perf_df.columns = ['1D', '1W', '1M', '3M', '6M', 'YTD']

#%% Streamlit

st.title('QIS Dashboard')

st.dataframe(perf_df.style.format("{:.2%}").background_gradient(cmap='coolwarm'))
st.line_chart(port_rets.loc['2023':].cumsum())
st.line_chart(offensive.loc['2023':].sum(axis=1).cumsum())
st.line_chart(interest.loc['2023':].sum(axis=1).cumsum())
st.line_chart(defensive.loc['2023':].sum(axis=1).cumsum())
