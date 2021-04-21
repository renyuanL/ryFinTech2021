# -*- coding: utf-8 -*-
"""
Created on Wed Apr 21 21:23:25 2021

@author: renyu
"""
import yahoo_fin
import yahoo_fin.stock_info as si
import pandas as pd 
from tqdm import tqdm 

sp500= si.tickers_sp500()

#%% 
theStats= {}
for t in tqdm(sp500):
    s= si.get_stats(t)
    #s= si.get_stats_valuation(t) 
    theStats[t]= s

#%%
theConcat= pd.concat(theStats, 
                     axis=1)

#%%

aL= [theStats[a]['Value'] for a in theStats.keys()]

bL= pd.concat(aL, axis=1)
bL.columns= theStats.keys()
bL.index=   theStats['A']['Attribute']
#%%
cL= bL.T
#%%
cL.to_excel('sp500_stats.xlsx')
#%%

