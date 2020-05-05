
# coding: utf-8

# In[6]:


from xgboost import XGBRegressor as XGBR 
from xgboost import  XGBClassifier as XGBC 
from sklearn.ensemble import RandomForestRegressor as RFR
from sklearn.linear_model import LinearRegression as LinearR 
from sklearn.datasets import load_boston
from sklearn.model_selection import KFold,cross_val_score as CVS,train_test_split as TTS
from sklearn.metrics import mean_squared_error as MSE
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt  
from sklearn.datasets import make_blobs
from sklearn.metrics import confusion_matrix as cm,recall_score as recall,roc_auc_score as auc


class_1 = 500 # 类别1 有500个样本
class_2 = 50 # 类别2 有50个样本

centers = [[0.0,0.0],[2.0,2.0]]  # 设定两个类别中心
clusters_std = [1.5,0.5] # 设定两个类别的方差，通常来说，样本量比较大的类别会比较松散

x,y = make_blobs(n_samples =[class_1,class_2],centers=centers,cluster_std=clusters_std,
                random_state =0,shuffle =False)

xtrain,xtest,ytrain,ytest = TTS(x,y,test_size =0.3,random_state = 420)

(y ==1).sum() /y.shape[0]


# In[11]:


x.shape


# In[12]:


y.shape


# In[13]:


clf = XGBC().fit(xtrain,ytrain)
ypred = clf.predict(xtest)


# In[14]:


ypred


# In[15]:


cm(ytest,ypred,labels=[1,0])


# In[16]:


recall(ytest,ypred)


# In[17]:


auc(ytest,clf.predict_proba(xtest)[:,1])


# In[18]:


# 正负样本比例
clf_ = XGBC(scale_pos_weight=10).fit(xtrain,ytrain)
ypred_ = clf_.predict(xtest)

clf_.score(xtest,ytest)


# 查看随着样本权重逐渐增加，模型的recall，auc和准确率如何变化

# In[19]:


for i in [1,5,10,20,30]:
    clf = XGBC(scale_pos_weight=i).fit(xtrain,ytrain)
    ypred_ = clf_.predict(xtest)
    print(i)
    print("\tAccuracy:{}".format(clf_.score(xtest,ytest)))
    print("\tRecall:{}".format(recall(ytest,ypred_)))
    print("\tAUC:{}".format(auc(ytest,clf_.predict_proba(xtest)[:,1])))


# # 使用xgboots本身自带的库

# In[21]:


import xgboost as xgb 

dtrain = xgb.DMatrix(xtrain,ytrain)
dtest = xgb.DMatrix(xtest,ytest)


# In[23]:


param = {"silent":True
          ,"obj":"binary:logistic"
          ,"eta":0.1
          ,"scale_pos_weight":1}

num_round = 100


# In[25]:


bst = xgb.train(param,dtrain,num_round)

preds = bst.predict(dtest)


# In[26]:


# 看看preds 返回了什么
preds 


# In[27]:


# 自己设定的阈值

ypred = preds.copy()
ypred[preds>0.5] =1 
ypred[ypred !=1] =0 


# In[31]:


# 写明参数

scale_pos_weight = [1,5,10]
names = ["negative vs positive: 1","negative vs positive:5","negative vs positive :10"]


# In[34]:


# 导入模型评估指标
from sklearn.metrics import accuracy_score as accuracy,recall_score as recall,roc_auc_score as auc

for name,i in zip(names,scale_pos_weight):
    param = {"silent":True
          ,"obj":"binary:logistic"
          ,"eta":0.1
          ,"scale_pos_weight":i}
    num_round = 100
    clf = xgb.train(param,dtrain,num_round)
    ypreds = clf.predict(dtest)
    ypred = preds.copy()
    ypred[preds>0.5] =1 
    ypred[ypred !=1] =0 
    print(name)
    print("\tAccuracy:{}".format(accuracy(ytest,ypred)))
    print("\tRecall:{}".format(recall(ytest,ypred)))
    print("\tAUC:{}".format(auc(ytest,preds)))
    

