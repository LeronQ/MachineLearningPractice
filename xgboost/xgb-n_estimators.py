
# coding: utf-8

# # 参数 n_estimators下的建模

# In[1]:


from xgboost import XGBRegressor as XGBR 
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


# In[2]:


# 导入数据集,数据集为字典形式
data = load_boston()
data


# In[3]:


x= data.data
y = data.target
print(x.shape)
print(y.shape)


# In[4]:


xtrain,xtest,ytrain,ytest = TTS(x,y,test_size =0.3,random_state = 420)


# In[5]:


reg = XGBR(n_estimators=100).fit(xtrain,ytrain)
reg.predict(xtest)


# In[6]:


# 测试集的结果分数,默认是返回R平方指标
reg.score(xtest,ytest)


# In[7]:


# 均方误差
MSE(ytest,reg.predict(xtest))


# In[8]:


y.mean()
# 均方误差结果大约占y均值的三分之一，效果一般


# In[9]:


# 树模型可以查看模型的重要性分数，可以使用嵌入法(select from model)进行特征选择
reg.feature_importances_


# # 使用交叉验证来进行对比

# In[10]:


reg = XGBR(n_estimators=100) # 交叉验证中导入没有经过训练的模型


# In[11]:


print(CVS(reg,xtrain,ytrain,cv=5))
# 1: mean 是对5次交叉验证求均值
# 2: 由于reg（XGB）默认是是R平方指标，所以交叉验证中也是返回R平方指标结果
CVS(reg,xtrain,ytrain,cv=5).mean()  


# In[12]:


CVS(reg,xtrain,ytrain,cv=5,scoring='neg_mean_squared_error').mean()


# In[13]:


# 来查看下sklearn中的所有模型来评估指标
import sklearn
sorted(sklearn.metrics.SCORERS.keys())


# In[14]:


# 使用随机森林和线性回归作为对比

#随机森林
rfr = RFR(n_estimators=100)
CVS(rfr,xtrain,ytrain,cv=5).mean()


# In[15]:


CVS(rfr,xtrain,ytrain,cv=5,scoring='neg_mean_squared_error').mean()


# In[16]:


#线性回归
lr = LinearR()
CVS(lr,xtrain,ytrain,cv=5,scoring='neg_mean_squared_error').mean()


# In[17]:


# 开启参数slient ,当数据巨大，训练缓慢的时候，可以使用这个参数来监控模型的进度

reg = XGBR(n_estimators=10,silent=False)
CVS(reg,xtrain,ytrain,cv=5,scoring='neg_mean_squared_error').mean()


# 学习曲线

# In[18]:


def plot_learning_curve(estimator, title, X, y,ax=None,
                        ylim=None, cv=None,n_jobs=None):
    from sklearn.model_selection import learning_curve
    import numpy as np
    import matplotlib.pyplot as plt
    train_sizes, train_scores, test_scores = learning_curve(estimator, X, y,shuffle=True,cv=cv,random_state=420,
                                                            n_jobs=n_jobs)
    
    if ax ==None:
        ax = plt.gca()
    else:
        ax = plt.figure()
    ax.set_title(title)
    if ylim is not None:
        ax.set_title(*ylim)
    ax.set_xlabel("Training examples")
    ax.set_ylabel("Score")
    ax.grid()
    ax.plot(train_sizes,np.mean(train_scores,axis=1),'o-',color='r',label="Training Score")
    ax.plot(train_sizes,np.mean(test_scores,axis=1),'o-',color='g',label="Test Score")
    ax.legend(loc="best")
    
    return ax 


# In[19]:


cv = KFold(n_splits=5,shuffle=True,random_state=42) # 交叉验证模式

plot_learning_curve(XGBR(n_estimators=100,random_state=420),"XGB",xtrain,ytrain,ax=None,cv=cv)
plt.show()


# 基于偏差的学习曲线

# In[20]:


axisx = range(10,1010,50)
rs = []

for i in axisx:
    reg = XGBR(n_estimators=i,random_state=420)
    rs.append(CVS(reg,xtrain,ytrain,cv=cv).mean())
print(axisx[rs.index(max(rs))],max(rs))
plt.figure(figsize=(20,5))
plt.plot(axisx,rs,c="red",label="XGB")
plt.legend()
plt.show()


# 基于方差和偏差的学习曲线

# In[21]:


axisx = range(50,500,50)
rs = []
var=[]
ge =[]
for i in axisx:
    reg = XGBR(n_estimators=i,random_state=420)
    cvresult =  CVS(reg,xtrain,ytrain,cv=cv)
    # 记录1- 偏差
    rs.append(cvresult.mean())
    # 记录方差
    var.append(cvresult.var())
    # 计算泛化误差的可控部分
    ge.append((1-cvresult.mean())**2+cvresult.var())

# 打印R2最高对应的参数取值，并打印这个参数下的方差
print(axisx[rs.index(max(rs))],max(rs),var[rs.index(max(rs))])

# 打印R2最低对应的参数取值，并打印这个参数下的R2
print(axisx[var.index(min(var))],rs[var.index(min(var))],min(var))

# # 打印泛化误差可控部分的参数取值，并打印这个参数下的R2，方差以及泛化误差的可控部分
print(axisx[ge.index(min(ge))],rs[ge.index(min(ge))],var[ge.index(min(ge))],min(ge))

plt.figure(figsize=(20,5))
plt.plot(axisx,rs,c="red",label="XGB")
plt.legend()
plt.show()

# 输出结果中，在150棵树的结果中表现最好


# In[22]:


axisx = range(100,300,10)
rs = []
var=[]
ge =[]
for i in axisx:
    reg = XGBR(n_estimators=i,random_state=420)
    cvresult =  CVS(reg,xtrain,ytrain,cv=cv)
    # 记录1- 偏差
    rs.append(cvresult.mean())
    # 记录方差
    var.append(cvresult.var())
    # 计算泛化误差的可控部分
    ge.append((1-cvresult.mean())**2+cvresult.var())

# 打印R2最高对应的参数取值，并打印这个参数下的方差
print(axisx[rs.index(max(rs))],max(rs),var[rs.index(max(rs))])

# 打印R2最低对应的参数取值，并打印这个参数下的R2
print(axisx[var.index(min(var))],rs[var.index(min(var))],min(var))

# # 打印泛化误差可控部分的参数取值，并打印这个参数下的R2，方差以及泛化误差的可控部分
print(axisx[ge.index(min(ge))],rs[ge.index(min(ge))],var[ge.index(min(ge))],min(ge))
rs = np.array(rs)
var = np.array(var)*0.01

plt.figure(figsize=(20,5))
plt.plot(axisx,rs,c="black",label="XGB")
# 添加方差线
plt.plot(axisx,rs+var,c="red",linestyle='-.')
plt.plot(axisx,rs-var,c="red",linestyle='-.')
plt.legend()
plt.show()


# 验证模型是否真的提高

# In[23]:


time0 = time()
print(XGBR(n_estimators=100,random_state=420).fit(xtrain,ytrain).score(xtest,ytest))
print(time()-time0)


# In[24]:


# 原始学习曲线表现的运行时间结果和score
time1= time()
print(XGBR(n_estimators=660,random_state=420).fit(xtrain,ytrain).score(xtest,ytest))
print(time()-time1)


# In[25]:


time2= time()
print(XGBR(n_estimators=180,random_state=420).fit(xtrain,ytrain).score(xtest,ytest))
print(time()-time2)

# 可以看到改进后的学习曲线运行效果较好


# # subsample参数

# In[26]:


# 在已经确定n_estimators=180时最好的情况下，查看采用比例多少合适

axisx = np.linspace(0,1,20)
rs = []

for i in axisx:
    reg = XGBR(n_estimators=180,subsample=i,random_state=420)
    rs.append(CVS(reg,xtrain,ytrain,cv=cv).mean())
print(axisx[rs.index(max(rs))],max(rs))
plt.figure(figsize=(20,5))
plt.plot(axisx,rs,c="red",label="XGB")
plt.legend()
plt.show()


# In[27]:


# 在已经确定n_estimators=180时最好的情况下，查看采用比例多少合适

axisx = np.linspace(0.05,1,20)
rs = []
var=[]
ge =[]
for i in axisx:
    reg = XGBR(n_estimators=180,subsample=i,random_state=420)
    cvresult =  CVS(reg,xtrain,ytrain,cv=cv)
    # 记录1- 偏差
    rs.append(cvresult.mean())
    # 记录方差
    var.append(cvresult.var())
    # 计算泛化误差的可控部分
    ge.append((1-cvresult.mean())**2+cvresult.var())

# 打印R2最高对应的参数取值，并打印这个参数下的方差
print(axisx[rs.index(max(rs))],max(rs),var[rs.index(max(rs))])

# 打印R2最低对应的参数取值，并打印这个参数下的R2
print(axisx[var.index(min(var))],rs[var.index(min(var))],min(var))

# # 打印泛化误差可控部分的参数取值，并打印这个参数下的R2，方差以及泛化误差的可控部分
print(axisx[ge.index(min(ge))],rs[ge.index(min(ge))],var[ge.index(min(ge))],min(ge))
rs = np.array(rs)
var = np.array(var)

plt.figure(figsize=(20,5))
plt.plot(axisx,rs,c="black",label="XGB")
# 添加方差线
plt.plot(axisx,rs+var,c="red",linestyle='-.')
plt.plot(axisx,rs-var,c="red",linestyle='-.')
plt.legend()
plt.show()


# In[28]:


# 进一步细化

# 在已经确定n_estimators=180时最好的情况下，查看采用比例多少合适

axisx = np.linspace(0.5,1,25)
rs = []
var=[]
ge =[]
for i in axisx:
    reg = XGBR(n_estimators=180,subsample=i,random_state=420)
    cvresult =  CVS(reg,xtrain,ytrain,cv=cv)
    # 记录1- 偏差
    rs.append(cvresult.mean())
    # 记录方差
    var.append(cvresult.var())
    # 计算泛化误差的可控部分
    ge.append((1-cvresult.mean())**2+cvresult.var())

# 打印R2最高对应的参数取值，并打印这个参数下的方差
print(axisx[rs.index(max(rs))],max(rs),var[rs.index(max(rs))])

# 打印R2最低对应的参数取值，并打印这个参数下的R2
print(axisx[var.index(min(var))],rs[var.index(min(var))],min(var))

# # 打印泛化误差可控部分的参数取值，并打印这个参数下的R2，方差以及泛化误差的可控部分
print(axisx[ge.index(min(ge))],rs[ge.index(min(ge))],var[ge.index(min(ge))],min(ge))
rs = np.array(rs)
var = np.array(var)

plt.figure(figsize=(20,5))
plt.plot(axisx,rs,c="black",label="XGB")
# 添加方差线
plt.plot(axisx,rs+var,c="red",linestyle='-.')
plt.plot(axisx,rs-var,c="red",linestyle='-.')
plt.legend()
plt.show()

# 即采样比例为0.625时表现最好


# In[29]:


reg = XGBR(n_estimators=180,subsample=0.65,random_state=420).fit(xtrain,ytrain)
reg.score(xtrain,ytrain)


# In[30]:


MSE(ytest,reg.predict(xtest))


# 由上面可以看到，由于数据量本身就很少，在进行采样后，模型效果反而降低了

# # 参数eta

# In[31]:


# 首先定义一个评分函数，这个评分函数能够帮助我们直接答应xtrain上的交叉验证结果

def regassess(reg,xtrain,ytrain,cv,scoring=['r2'],show=True):
    score = []
    for i in range(len(scoring)):
        if show:
            print("{}:{:.2f}".format(scoring[i],CVS(reg,xtrain,ytrain,cv=cv,scoring=scoring[i]).mean()))
        score.append(CVS(reg,xtrain,ytrain,cv=cv,scoring=scoring[i]).mean()) 
    return score

reg = XGBR(n_estimators=180,random_state=420).fit(xtrain,ytrain)                
                  


# In[32]:


regassess(reg,xtrain,ytrain,cv,scoring=["r2","neg_mean_squared_error"])


# In[33]:


regassess(reg,xtrain,ytrain,cv,scoring=["r2","neg_mean_squared_error"],show=False)


# In[34]:


from time import  time
import datetime
for i in [0,0.2,0.5,1]:
    reg = XGBR(n_estimators=180,random_state=420,learning_rate=i)
    print("learning_rate={}".format(i))
    regassess(reg,xtrain,ytrain,cv,scoring=["r2","neg_mean_squared_error"])
    print("\t")


# # 网格搜索

# 网格搜索不一定是最好的参数，往往运行的结果也不具有数学解释意义，只供参数选择的一种参考

# In[43]:


data = load_boston()
x= data.data
y = data.target
xtrain,xtest,ytrain,ytest = TTS(x,y,test_size =0.3,random_state = 420)


# In[49]:


from sklearn.model_selection import GridSearchCV

param = {"reg_alpha":np.arange(0,5,0.1),"reg_lambda":np.arange(0.,2,0.1)}

gscv = GridSearchCV(reg,param_grid = param, scoring ="neg_mean_squared_error",cv=cv)

time0 = time()
gscv.fit(xtrain,ytrain)

preds = gscv.predict(xtest)

from sklearn.metrics import r2_score,mean_squared_error as MSE

print("r2_score:",r2_score(ytest,preds))
print("MSE:",MSE(ytest,preds))


# In[48]:





# In[50]:


gscv.best_estimator_,gscv.best_params_


# # 让树停止生长，参数gamma与xgb.cb

# 基于gamma参数的学习曲线

# In[51]:


axisx = np.arange(0,5,0.05)
rs = []
var=[]
ge =[]
for i in axisx:
    reg = XGBR(n_estimators=180,random_state=420,gamma=i)
    cvresult =  CVS(reg,xtrain,ytrain,cv=cv)
    # 记录1- 偏差
    rs.append(cvresult.mean())
    # 记录方差
    var.append(cvresult.var())
    # 计算泛化误差的可控部分
    ge.append((1-cvresult.mean())**2+cvresult.var())

# 打印R2最高对应的参数取值，并打印这个参数下的方差
print(axisx[rs.index(max(rs))],max(rs),var[rs.index(max(rs))])

# 打印R2最低对应的参数取值，并打印这个参数下的R2
print(axisx[var.index(min(var))],rs[var.index(min(var))],min(var))

# # 打印泛化误差可控部分的参数取值，并打印这个参数下的R2，方差以及泛化误差的可控部分
print(axisx[ge.index(min(ge))],rs[ge.index(min(ge))],var[ge.index(min(ge))],min(ge))
rs = np.array(rs)
var = np.array(var)*0.1

plt.figure(figsize=(20,5))
plt.plot(axisx,rs,c="black",label="XGB")
# 添加方差线
plt.plot(axisx,rs+var,c="red",linestyle='-.')
plt.plot(axisx,rs-var,c="red",linestyle='-.')
plt.legend()
plt.show()


# In[55]:


import xgboost as xgb 

dfull = xgb.DMatrix(x,y)

# 设定参数,默认评估指标是rmse
param1 = {"silent":True,"obj":"reg:linear","ganmma":0}
num_round = 180
n_fold = 5

# 使用xgb.cv
curesult = xgb.cv(param1,dfull,num_round,n_fold)


# In[57]:


#查看xgb.cv生成的结果
# 可以看到，随着树的增加，模型效果在一直变化
curesult


# In[64]:


plt.figure(figsize=(25,5))
plt.grid()
plt.plot(range(1,181),curesult.iloc[:,0],c="red",label="train,gamma=0")
plt.plot(range(1,181),curesult.iloc[:,2],c="blue",label="test,gamma=0")
plt.legend()
plt.show()


# 自定义评估指标，修改为mae

# In[66]:


param1 = {"silent":True,"obj":"reg:linear","ganmma":0,"eval_metric":"mae"} # 修改为mae
num_round = 180
n_fold = 5

# 使用xgb.cv
curesult = xgb.cv(param1,dfull,num_round,n_fold)

plt.figure(figsize=(25,5))
plt.grid()
plt.plot(range(1,181),curesult.iloc[:,0],c="red",label="train,gamma=0")
plt.plot(range(1,181),curesult.iloc[:,2],c="blue",label="test,gamma=0")
plt.legend()
plt.show()


# 由上图可以看到，随着树增加，训练集和测试集误差不再变化，有过拟合的迹象

# 增加惩罚项

# In[70]:


import xgboost as xgb 

dfull = xgb.DMatrix(x,y)

param1 = {"silent":True,"obj":"reg:linear","ganmma":0}
param2 = {"silent":True,"obj":"reg:linear","ganmma":20}
num_round = 180
n_fold = 5

# 使用xgb.cv
curesult1 = xgb.cv(param1,dfull,num_round,n_fold)

curesult2 = xgb.cv(param2,dfull,num_round,n_fold)


plt.figure(figsize=(25,10))
plt.grid()
plt.plot(range(1,181),curesult1.iloc[:,0],c="red",label="train,gamma=0")
plt.plot(range(1,181),curesult1.iloc[:,2],c="blue",label="test,gamma=0")
plt.plot(range(1,181),curesult2.iloc[:,0],c="green",label="train,gamma=20")
plt.plot(range(1,181),curesult2.iloc[:,2],c="black",label="test,gamma=20")
plt.legend()
plt.show()

# gamma是通过控制训练集上的训练，来降低训练集的表现，从而降低与测试集上的差距


# # 分类数据集，来查看gamma参数影响

# In[92]:


from sklearn.datasets import load_breast_cancer

data2 = load_breast_cancer()
x2 = data.data
y2 = data.target

dfull2 = xgb.DMatrix(x2,y2)

param1 ={"silent":True,"obj":"binary:logistic","ganmma":0,"nfold":5}
param2 ={"silent":True,"obj":"binary:logistic","ganmma":2,"nfold":5}

num_round = 100
curesult1 = xgb.cv(param1,dfull2,num_round)
curesult2 = xgb.cv(param2,dfull2,num_round)

plt.figure(figsize=(25,10))
plt.grid()
plt.plot(range(1,101),curesult1.iloc[:,0],c="red",label="train,gamma=0")
plt.plot(range(1,101),curesult1.iloc[:,2],c="orange",label="test,gamma=0")
# plt.plot(range(1,101),curesult2.iloc[:,0],c="green",label="train,gamma=2")
# plt.plot(range(1,101),curesult2.iloc[:,2],c="blue",label="test,gamma=2")
plt.legend()
plt.show()


# In[83]:


plt.figure(figsize=(25,10))
plt.grid()
# plt.plot(range(1,101),curesult1.iloc[:,0],c="red",label="train,gamma=0")
# plt.plot(range(1,101),curesult1.iloc[:,2],c="orange",label="test,gamma=0")
plt.plot(range(1,101),curesult2.iloc[:,0],c="green",label="train,gamma=2")
plt.plot(range(1,101),curesult2.iloc[:,2],c="blue",label="test,gamma=2")
plt.legend()
plt.show()


# # 过拟合:使用xgb.cv剪枝参数与回归模型调参
# 

# 作为天生过拟合的模型, XGBoost应用的核心之一就是减轻过拟合带来的影响。作为树模型,就是减轻过拟合的方式主
# 要是靠对决策树剪枝来降低模型的复杂度,以求降低方差。防止
# 过拟合的参数,包括复杂度控制，正则化的两个参数λ和x,控制送代速度的参数以及管理每次迭
# 代前进行的随机有放回抽样的参数 subsample,所有的这些参数都可以用来减轻过拟合。但除此之外,我们还有
# 几个影响重大的,专用于剪枝的参数:（1）树的最大深度；（2）每次生成树随机抽样特征的比例；（3）每次生成树的一层时，随机抽样特征的比例；（4）每次叶子节点时，随机抽样特征的比例；（5）一个叶子节点所需的最小hi，即叶子节点上的二阶导数之和，类似于样本权重。其中树的最大深度设置影响最大
# 

# In[93]:


data2 = load_breast_cancer()
x2 = data.data
y2 = data.target

dfull = xgb.DMatrix(x2,y2)


# In[104]:


import matplotlib.pyplot as plt 
plt.rc('font', family='SimHei', size=15) 

param1 = {"silent":True
          ,"obj":"binary:logistic"
          ,"subsample":1
          ,"max_depth":6
          ,"eta":0.3
          ,"lambda":1
          ,"ganmma":0
          ,"alpha":1
          ,"colsample_bytree":1
          ,"colsample_bylevel":1
          ,"colsample_bylevel":1
          ,"nfold":5}

num_round = 200
n_fold = 5

# 使用xgb.cv
curesult = xgb.cv(param1,dfull,num_round,n_fold)

plt.figure(figsize=(25,10))
plt.grid()
plt.plot(range(1,201),curesult.iloc[:,0],c="red",label="train,original")
plt.plot(range(1,201),curesult.iloc[:,2],c="orange",label="test,original")
plt.xlabel("树的数目")
plt.ylabel("训练误差")
plt.legend()
plt.show()


# In[109]:


import matplotlib.pyplot as plt 
plt.rc('font', family='SimHei', size=15) 

param1 = {"silent":True
          ,"obj":"binary:logistic"
          ,"subsample":1
          ,"max_depth":6
          ,"eta":0.3
          ,"lambda":1
          ,"ganmma":0
          ,"alpha":1
          ,"colsample_bytree":1
          ,"colsample_bylevel":1
          ,"colsample_bylevel":1
          ,"nfold":5}

param2 = {"silent":True
          ,"obj":"binary:logistic"
          ,"max_depth":2
          ,"nfold":5}


num_round = 200
n_fold = 5

# 使用xgb.cv
curesult1 = xgb.cv(param1,dfull,num_round,n_fold)
curesult2 = xgb.cv(param2,dfull,num_round,n_fold)




plt.figure(figsize=(25,10))
plt.grid()
plt.plot(range(1,201),curesult1.iloc[:,0],c="red",label="train,original")
plt.plot(range(1,201),curesult1.iloc[:,2],c="orange",label="test,original")
plt.plot(range(1,201),curesult2.iloc[:,0],c="green",label="train,last")
plt.plot(range(1,201),curesult2.iloc[:,2],c="blue",label="test,last")
plt.xlabel("树的数目")
plt.ylabel("训练误差")
plt.legend()
plt.show()

# 可以看到训练表现提升，测试表现下降，模型有明显的过拟合改善效果

