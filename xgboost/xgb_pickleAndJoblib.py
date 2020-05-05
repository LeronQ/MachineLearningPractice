
# coding: utf-8

# 使用pickle和joblib保存和调用模型

# # 使用pickle保存模型

# In[1]:


import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor as RFR
from sklearn.linear_model import LinearRegression as LinearR 
from sklearn.datasets import load_boston
from sklearn.model_selection import KFold,cross_val_score as CVS,train_test_split as TTS
from sklearn.metrics import mean_squared_error as MSE
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
from time import time 
import datetime
import pickle

data = load_boston()
x = data.data
y = data.target

dfull = xgb.DMatrix(x,y)

param1 = {"silent":True
          ,"obj":"binary:linear"
          ,"subsample":1
          ,"max_depth":4
          ,"eta":0.05
          ,"lambda":1 
          ,"ganmma":20
          ,"alpha":0.2
          ,"colsample_bytree":0.4
          ,"colsample_bylevel":0.6
          ,"colsample_bylevel":1
          ,"nfold":5}

num_round = 200
bst = xgb.train(param1,dfull,num_round)

# 保存模型

pickle.dump(bst,open("xgboost_boston.dat","wb"))

# 注意，open中往往使用w或者r作为读取的模式，但其实w与r只能用于文本文件，当我们希望导入的不是
# 不是文本文件而是模型本身的时候，我们使用wb或者rb作为读取的模式
# 其中wb表示以二进制写入，rb表示以二进制读入，使用open进行保存的这个文件中是一个可以进行
# 读取或者调用的模型

import sys
sys.path


# In[3]:


xtrain,xtest,ytrain,ytest = TTS(x,y,test_size =0.3,random_state = 420)


# In[4]:


dtest = xgb.DMatrix(xtest,ytest)


# In[7]:


load_model = pickle.load(open("xgboost_boston.dat","rb"))
                              
print("load model from：xgboost_boston.dat")
                              


# In[9]:


ypreds = load_model.predict(dtest)

MSE(ytest,ypreds)


# # 使用joblib保存模型

# joblib是scipy生态系统中的一部分，它为python提供保存模型的功能，处理numpy结构数据非常高效，对很大的数据集和巨大的模型非常有用，

# In[11]:


bst = xgb.train(param1,dfull,num_round)

import joblib

joblib.dump(bst,"xgboost_boston2.dat")

load_model2 = joblib.load("xgboost_boston2.dat")

ypreds2 = load_model2.predict(dtest)

MSE(ytest,ypreds2)

