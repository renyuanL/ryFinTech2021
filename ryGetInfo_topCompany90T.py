# -*- coding: utf-8 -*-
"""
投資股票就是買公司成為公司的股東，
那麼，全世界的 好(或大)公司 在哪裡？
先看看

美國、
日本、
台灣、
中國、

有哪些上得了檯面的公司。

Created on Wed Apr 28 20:50:16 2021

@author: renyu
"""


# read_html from the following urls
'''
https://companiesmarketcap.com/
https://companiesmarketcap.com/page/2/
https://companiesmarketcap.com/page/3/
:
https://companiesmarketcap.com/page/49/
'''

# to get the info about ...
'''
Largest Companies by Market Cap
companies: 4,840     
total market cap: $90.176 T
'''

import pandas as pd
from tqdm import tqdm


aUrl= 'https://companiesmarketcap.com/'
print(f'.. get info from {aUrl}')

aTbl= pd.read_html(aUrl)
topComps= aTbl[0]


aD= {'p1':topComps}
for i in tqdm(range(2,50)):
    
    url=     f'{aUrl}page/{i}/'
    print(f'.. get info from {url}')
    
    tbl= pd.read_html(url)
    topComps= tbl[0]
    aD[f'p{i}']= topComps
    

aDF1= pd.concat(aD)

fp= pd.ExcelWriter('topCompany90T.xlsx')
    
aDF1.to_excel(fp, sheet_name= 'MarketCap')

#%%

aUrl= 'https://companiesmarketcap.com/assets-by-market-cap/'
aTbl= pd.read_html(aUrl)[0]
#%%

'''
https://companiesmarketcap.com/most-profitable-companies/

https://companiesmarketcap.com/most-profitable-companies/page/2/

https://companiesmarketcap.com/most-profitable-companies/page/49/
'''

aUrl= 'https://companiesmarketcap.com/most-profitable-companies/'
print(f'.. get info from {aUrl}')

aTbl= pd.read_html(aUrl)
topComps= aTbl[0]


aD= {'p1':topComps}
for i in tqdm(range(2,50)):
    
    url=     f'{aUrl}page/{i}/'
    print(f'.. get info from {url}')
    
    tbl= pd.read_html(url)
    topComps= tbl[0]
    aD[f'p{i}']= topComps
    

aDF2= pd.concat(aD)

aDF2.to_excel(fp, sheet_name= 'Earnings')

fp.close()
#%%
