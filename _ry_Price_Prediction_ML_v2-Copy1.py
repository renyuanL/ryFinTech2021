'''


"Machine Learning for investing in stocks — a comparison of different ML models for predicting the returns of the SPY"

by Ferhat Culfaz
2021/Mar/23

文章： http://medium.datadriveninvestor.com/machine-learning-for-investing-in-stocks-a-comparison-of-different-ml-models-for-predicting-the-ec7a7e4f236d

程式碼: 
    https://github.com/ferhat00/PortfolioOptimisation

單支：
https://github.com/ferhat00/PortfolioOptimisation/blob/main/Price_Prediction_ML_v2.ipynb

原單支程式碼有小bug，先改一份可跑，原理尚未理解，裡面有很多投資、技術指標專有名詞尚待研究釐清。
很好的學習材料，可作為碩論的入門磚。
'''

#!/usr/bin/env python
# coding: utf-8

# In[1]:


from datetime import date
today = date.today()


# In[2]:


#stock = ['SPY', 'GLD', 'SPLB', 'EEM', 'QQQ', 'SLV', 'HYG', 'VWO', 'TLT', 'FXI']

#### _ry_modified
#### stock = ['SPY']

theStockName= 'SPY' #'^GSPC' #'AMZN' #'MSFT' #'AAPL'

stock = [theStockName]
####################

sampling= ['daily', 'weekly'][0]
aggregation= ['last_day', 'mean'][0]
#stock = ['VTI', 'VEU', 'VNQ', 'BND', 'GSG']

#### _ry_modified
#date_start = '1993-01-01'
date_start = '1980-01-01'

date_end = today.strftime("%Y-%m-%d")

# _ry_註解
# training/testing 界線日期
#
cutoff =  '2010-01-01' #'2012-12-31'


# In[3]:


import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt
import mplfinance as mpf
import pandas_datareader as pdr
import statsmodels.api as sm
import scipy.stats as scs


# In[4]:


combined_data = []
for i in range(len(stock)):
    df = pdr.DataReader(stock[i], 
                       start=       date_start, 
                       end=         date_end, 
                       data_source= 'yahoo')
    name = stock[i]
    df_stock = df[['Close']] #df[['Adj Close']]
    df_stock = df_stock.rename(columns={"Close" : name})
    if i > 0:
        combined_data = pd.concat([combined_data, df_stock], axis = 1)
    else:
        combined_data = df_stock
        
    
    #combined_data.append(df_stock)
    #combined_data.append(df_stock)
    #combined_data = pd.concat([combined_data, df_stock], axis=1, join="inner" )
    #df1.append(df4, ignore_index=True, sort=False)
    #result = pd.concat([df1, df4], axis=1, join="inner")
if sampling == 'weekly':
    if aggregation == 'last_day':
        combined_data = combined_data.resample('W').agg('last')
    elif aggregation == "mean":
        combined_data = combined_data.resample('W').agg('mean')


# In[5]:


combined_data


# In[6]:


(combined_data / combined_data.iloc[0] * 100).plot(figsize=(20, 12))


# In[7]:


combined_data['return'] = np.log(combined_data / combined_data.shift(1))
combined_data['direction'] = np.where(combined_data['return'] > 0, 1, 0)
combined_data.dropna(inplace=True)
combined_data.head()


# In[8]:


combined_data['return'].hist(bins=50, figsize=(10, 8));


# # Feature Engineering

# In[9]:


import ta
from ta.volatility import BollingerBands


# In[10]:


lags = 5
cols = []
for lag in range(1, lags + 1):
    col = 'lag_{}'.format(lag)
    combined_data[col] = combined_data['return'].shift(lag)
    cols.append(col)
combined_data.dropna(inplace=True)


# In[11]:


indicator_bb = BollingerBands(close=combined_data[theStockName], window=20, window_dev=2)


bb = pd.DataFrame()
bb['bb_bbm'] = indicator_bb.bollinger_mavg()
bb['bb_bbh'] = indicator_bb.bollinger_hband()
bb['bb_bbl'] = indicator_bb.bollinger_lband()

mini, maxi = bb['bb_bbm'].min(), bb['bb_bbm'].max()
bb['bb_bbm_norm'] = (bb['bb_bbm'] - mini) / (maxi - mini)

mini, maxi = bb['bb_bbm'].min(), bb['bb_bbh'].max()
bb['bb_bbh_norm'] = (bb['bb_bbh'] - mini) / (maxi - mini)

mini, maxi = bb['bb_bbl'].min(), bb['bb_bbl'].max()
bb['bb_bbl_norm'] = (bb['bb_bbl'] - mini) / (maxi - mini)


# In[12]:


from ta.trend import MACD


# In[13]:


indicator_macd = MACD(close = combined_data[theStockName], window_slow = 26, window_fast = 12, window_sign = 9, fillna = False)
macd = pd.DataFrame()
macd['macd'] = indicator_macd.macd()
macd['macd_diff'] = indicator_macd.macd_diff()
macd['macd_signal'] = indicator_macd.macd_signal()

mini, maxi = macd['macd'].min(), macd['macd'].max()
macd['macd_norm'] = (macd['macd'] - maxi) / (maxi - mini)

mini, maxi = macd['macd_diff'].min(), macd['macd_diff'].max()
macd['macd_diff_norm'] = (macd['macd_diff'] - maxi) / (maxi - mini)

mini, maxi = macd['macd_signal'].min(), macd['macd_signal'].max()
macd['macd_signal_norm'] = (macd['macd_signal'] - maxi) / (maxi - mini)


# In[14]:


normalised_features = pd.merge(bb[['bb_bbm_norm', 'bb_bbh_norm', 'bb_bbl_norm']],macd[['macd_norm','macd_diff_norm', 'macd_signal_norm']], on='Date', how = 'left')


# In[15]:


normalised_features


# In[16]:


combined_data = pd.merge(combined_data, normalised_features, on='Date', how = 'left')


# In[17]:


combined_data


# In[18]:


combined_data['momentum'] = combined_data['return'].rolling(5).mean().shift(1)
combined_data['volatility'] = combined_data['return'].rolling(20).std().shift(1)
#combined_data['distance'] = (combined_data[stock] - combined_data[stock].rolling(50).mean()).shift(1)
combined_data.dropna(inplace=True)
combined_data


# In[19]:


combined_data.plot(figsize=(20, 12))


# # Which columns to use for training

# In[20]:


cols = list(combined_data.columns)


# In[21]:


cols = cols[3:]
cols


# # Linear Regression

# In[22]:


from sklearn.metrics import accuracy_score
from sklearn import linear_model
from sklearn import preprocessing
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score


# In[23]:


combined_data_linear = combined_data.copy()
combined_data_linear


# In[24]:


lim = linear_model.LinearRegression()


# In[25]:


X = combined_data_linear[cols]
y = combined_data_linear['return']


# In[28]:


#
# _ry_ debug...
#
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, 
        random_state=42)


# In[29]:


cv = cross_val_score(lim, X_test, y_test, cv=10)
cv.mean()


# In[30]:


lim.fit(X, y)


# In[31]:


combined_data_linear['pos_ols_1'] = lim.fit(combined_data_linear[cols], combined_data_linear['return']).predict(combined_data_linear[cols])
combined_data_linear['pos_ols_2'] = lim.fit(combined_data_linear[cols], combined_data_linear['direction']).predict(combined_data_linear[cols])


# In[32]:


combined_data_linear[['pos_ols_1']] = np.where(combined_data_linear[['pos_ols_1']] > 0, 1, -1)
combined_data_linear[['pos_ols_2']] = np.where(combined_data_linear[['pos_ols_2']] > 0, 1, -1)


# In[33]:


combined_data_linear['pos_ols_1'].value_counts()


# In[34]:


combined_data_linear['pos_ols_2'].value_counts()


# Number of trades

# In[35]:


(combined_data_linear['pos_ols_1'].diff() != 0).sum()


# In[36]:


(combined_data_linear['pos_ols_2'].diff() != 0).sum()


# In[37]:


combined_data_linear


# In[38]:


combined_data_linear['strat_ols_1'] = combined_data_linear['return']*combined_data_linear['pos_ols_1']
combined_data_linear['strat_ols_2'] = combined_data_linear['return']*combined_data_linear['pos_ols_2']


# In[39]:


accuracy_score(combined_data_linear['pos_ols_1'],
               np.sign(combined_data_linear['return']))


# In[40]:


accuracy_score(combined_data_linear['pos_ols_2'],
               np.sign(combined_data_linear['return']))


# In[41]:


combined_data_linear[['return', 'strat_ols_1', 'strat_ols_2']].sum().apply(np.exp)


# In[42]:


ax = combined_data_linear[['return', 'strat_ols_1', 'strat_ols_2']].cumsum().apply(np.exp).plot(
                                        figsize=(20, 8));
combined_data_linear['pos_ols_1'].plot(ax = ax, lw=1.5, secondary_y = 'Position', style = '--')
combined_data_linear['pos_ols_2'].plot(ax = ax, lw=1.5, secondary_y = 'Position', style = '--')
ax.get_legend().set_bbox_to_anchor((0.25,0.85));


# # Logistic Regression

# In[43]:


combined_data_logistic = combined_data.copy()


# In[44]:


lm = linear_model.LogisticRegression(C=1e7, solver='lbfgs',
                                     multi_class='auto',
                                     max_iter=10000)


# In[45]:


lm.fit(combined_data_logistic[cols], np.sign(combined_data_logistic['return']))


# In[46]:


combined_data_logistic['prediction'] = lm.predict(combined_data_logistic[cols])


# In[47]:


combined_data_logistic['prediction'].value_counts()


# In[48]:


hits = np.sign(combined_data_logistic['return'].iloc[lags:] *
               combined_data_logistic['prediction'].iloc[lags:]
              ).value_counts()
hits


# In[49]:


accuracy_score(combined_data_logistic['prediction'],
               np.sign(combined_data_logistic['return']))


# In[50]:


combined_data_logistic['strategy'] = combined_data_logistic['return']*combined_data_logistic['prediction'].shift(1)


# In[51]:


combined_data_logistic[['return', 'strategy']].sum().apply(np.exp)


# In[52]:


combined_data_logistic


# In[53]:


ax = combined_data_logistic[['return', 'strategy']].cumsum().apply(np.exp).plot(
                                        figsize=(20, 8));
combined_data_logistic['prediction'].plot(ax = ax, lw=1.5, secondary_y = 'Position', style = '--')
ax.get_legend().set_bbox_to_anchor((0.25,0.85));


# # Lets avoid overfitting

# In[54]:


from sklearn.model_selection import train_test_split


# In[55]:


train, test = train_test_split(combined_data_logistic, test_size=0.5,
                               shuffle=False, random_state=100)


# In[56]:


train = train.copy().sort_index()
test = test.copy().sort_index()  


# In[57]:


lm.fit(train[cols], np.sign(train['return']))


# In[58]:


test['prediction'] = lm.predict(test[cols])


# In[59]:


test['prediction'].value_counts()


# In[60]:


hits = np.sign(test['return'].iloc[lags:] *
               test['prediction'].iloc[lags:]
              ).value_counts()
hits


# In[61]:


accuracy_score(test['prediction'],
               np.sign(test['return']))


# In[62]:


test['strategy'] = test['return']*test['prediction'].shift(1)


# In[63]:


test[['return', 'strategy']].sum().apply(np.exp)


# In[64]:


ax = test[['return', 'strategy']].cumsum().apply(np.exp).plot(
                                        figsize=(20, 8));
test['prediction'].plot(ax = ax, lw=1.5, secondary_y = 'Position', style = '--')
ax.get_legend().set_bbox_to_anchor((0.25,0.85));


# # K-Fold Cross Validation

# In[65]:


k = 4
num_val_samples = len(combined_data_logistic) // k

all_scores = []
for i in range(k):
    print('processing fold #', i)
    # Prepare the validation data: data from partition # k
    val_data = combined_data_logistic[cols][i * num_val_samples: (i + 1) * num_val_samples]
    val_targets = np.sign(combined_data_logistic['return'])[i * num_val_samples: (i + 1) * num_val_samples]

    # Prepare the training data: data from all other partitions
    partial_train_data = np.concatenate(
        [combined_data_logistic[cols][:i * num_val_samples],
         combined_data_logistic[cols][(i + 1) * num_val_samples:]],
        axis=0)
    partial_train_targets = np.concatenate(
        [np.sign(combined_data_logistic['return'])[:i * num_val_samples],
         np.sign(combined_data_logistic['return'])[(i + 1) * num_val_samples:]],
        axis=0)

    # Build the Keras model (already compiled)
    lm = linear_model.LogisticRegression(C=1e7, solver='lbfgs',
                                     multi_class='auto',
                                     max_iter=1000)
    # Train the model (in silent mode, verbose=0)
    lm.fit(partial_train_data, np.sign(partial_train_targets))
    # Evaluate the model on the validation data
    val_data['prediction'] = lm.predict(val_data)
    val_data['return'] =  combined_data_logistic['return'][i * num_val_samples: (i + 1) * num_val_samples]
    ac = accuracy_score(val_data['prediction'], np.sign(val_data['return']))
    all_scores.append(ac)


# In[66]:


all_scores


# # Clustering

# In[67]:


combined_data_logistic


# In[68]:


from sklearn.cluster import KMeans


# In[69]:


k_col = cols[0:5]
k_col


# In[70]:


model = KMeans(algorithm = 'auto', copy_x = True, init = 'k-means++', max_iter = 300, n_clusters=3, n_init = 10, random_state=0, tol=0.0001, verbose=0)  #  <1>


# In[71]:


model.fit(combined_data_logistic[k_col])


# In[72]:


combined_data_logistic['pos_clus'] = model.predict(combined_data_logistic[k_col])


# In[73]:


combined_data_logistic['pos_clus'] = np.where(combined_data_logistic['pos_clus'] == 2, -1, 1) 


# In[74]:


combined_data_logistic['pos_clus'].values


# In[75]:


plt.figure(figsize=(10, 6))
plt.scatter(combined_data_logistic[cols].iloc[:, 0], combined_data_logistic[cols].iloc[:, 1],
            c=combined_data_logistic['pos_clus'], cmap='coolwarm');


# In[76]:


combined_data_logistic['strat_clus'] = combined_data_logistic['return']*combined_data_logistic['pos_clus']


# In[77]:


combined_data_logistic[['return', 'strat_clus']].sum().apply(np.exp)


# In[78]:


combined_data_logistic[['return', 'strat_clus']].cumsum().apply(np.exp).plot(figsize=(10, 6));


# # Neural Network Keras

# In[79]:


import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import random


# In[80]:


optimizer = Adam(learning_rate=0.0001)


# In[81]:


def set_seeds(seed=100):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(100)


# In[82]:


metric = 'accuracy'


# In[83]:


set_seeds()
model = Sequential()
model.add(Dense(64, activation='relu',
        input_shape=(len(cols),)))
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid')) # <5>
model.compile(optimizer=optimizer,
              loss='binary_crossentropy',
              metrics=[metric])


# In[84]:

# _ry_註解
# training/testing 界線日期
#
# cutoff =  '2000-01-01' #'2012-12-31'


# In[85]:


combined_data['direction'] = np.where(combined_data['return'] > 0, 1, 0)
combined_data


# In[86]:


training_data = combined_data[combined_data.index < cutoff].copy()


# In[87]:


mu, std = training_data.mean(), training_data.std()


# In[88]:


training_data_ = (training_data - mu) / std
training_data_


# In[89]:


test_data = combined_data[combined_data.index >= cutoff].copy()


# In[90]:


#test_data_ = (test_data - mu) / std


# In[91]:

'''
get_ipython().run_cell_magic('time', '', "model.fit(training_data[cols],\n          training_data['direction'],\n          epochs=50, verbose=False,\n          validation_split=0.2, shuffle=False)")
'''
model.fit(training_data[cols],
          training_data['direction'],
          epochs= 100, #50, 
          verbose=False,
          validation_split=0.2, shuffle=False)

# In[92]:


res= pd.DataFrame(model.history.history)


# In[93]:


res.head()


# In[94]:


res[[metric, 'val_' + metric]].plot(figsize=(10, 6), style='--');


# In[95]:


model.evaluate(training_data_[cols], training_data['direction'])


# In[96]:


pred= model.predict_classes(training_data_[cols])


# In[97]:


pred[:30].flatten()


# In[98]:


training_data['prediction'] = np.where(pred > 0, 1, -1)


# In[99]:


training_data['strategy'] = (training_data['prediction'] *
                            training_data['return'])


# In[100]:


training_data[['return', 'strategy']].sum().apply(np.exp)


# In[101]:


training_data[['return', 'strategy']].cumsum(
                ).apply(np.exp).plot(figsize=(10, 6));


# In[104]:


#
# _ry_debug ...
#
#model.evaluate(test_data_[cols], test_data['direction'])
model.evaluate(test_data[cols], test_data['direction'])


# In[105]:


cols


# In[108]:


#
# _ry_debug ...
#

#pred = model.predict_classes(test_data_[cols])
pred = model.predict_classes(test_data[cols])
pred


# In[109]:


test_data['prediction'] = np.where(pred > 0, 1, -1)


# In[110]:


test_data['prediction'].value_counts()


# In[111]:


test_data['strategy'] = (test_data['prediction'] *
                        test_data['return'])


# In[112]:


test_data[['return', 'strategy']].sum().apply(np.exp)


# In[113]:


ax = test_data[['return', 'strategy']].cumsum(
                ).apply(np.exp).plot(figsize=(10, 6));
test_data['prediction'].plot(ax = ax, lw=1, secondary_y = 'Position', style = '--')
ax.get_legend().set_bbox_to_anchor((0.25,0.85));


# In[ ]:




