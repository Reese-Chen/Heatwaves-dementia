#%% 读入数据
import numpy as np
import pandas as pd

prev = pd.read_csv(r'd:\\heatwave and dementia\\data\\prevalence of dementia(1).csv',
                   sep=',',header='infer')
weight = pd.read_csv(r'd:\\heatwave and dementia\\data\\各国ICA.csv',
                   sep=',',header='infer')
ica_hw = np.loadtxt(r'd:\\heatwave and dementia\\data\\ica_hw_with_trend.csv',
               delimiter=',')

#%% 数据处理

# Y的获取
prev1 = prev.where((prev.sex_id==3)&(prev.age_id==22)).dropna()
countryname = list(prev1.location_name)
countryname = list(set(countryname))
countryname.sort()
for i,cname in enumerate(countryname):
    prev2 = prev1["val"].where(prev1.location_name==cname).dropna()
    if i==0:
        Y = list(prev2)
    else:
        Y = np.c_[Y,list(prev2)]

# W的获取
weight = weight.iloc[:,1:]
W = weight.values

# X的获取
X = np.zeros((30,40))
for i in range(30):
    for j in range(40):
        X[i,j] = np.sum(ica_hw[i*365:(i+1)*365,j])
        
#%% 数据保存
np.savetxt('d:\\heatwave and dementia\\data\\Y_prevalence30年不分类数据(trend).csv',
           Y,delimiter = ',')
np.savetxt('d:\\heatwave and dementia\\data\\W_各国对ICA权重数据(trend).csv',
           W,delimiter = ',')
np.savetxt('d:\\heatwave and dementia\\data\\X_ICA30年数据(trend).csv',
           X,delimiter = ',')

#%% 数据格式转换

m = 161
n = 40
t = 30

#数据格式转换
for i in range(m):
    if i==0:
        Y1 = Y[:,i]
        X1 = X*W[i,:]
    else:
        Y1 = np.r_[Y1,Y[:,i]]
        X1 = np.r_[X1,X*W[i,:]]
        
#%% 基本数据展示
import matplotlib.pyplot as plt


import random
def randomcolor():
    colorArr = ['1','2','3','4','5','6','7','8','9','A','B','C','D','E','F']
    color = ""
    for i in range(6):
        color += colorArr[random.randint(0,14)]
    return "#"+color

# 各自变化趋势
for i in range(161):
    plt.plot(Y[:,i],color=randomcolor())
plt.show()
for i in range(40):
    plt.plot(X[:,i],color=randomcolor())
plt.show()

# prevalence 和ICA时间序列的关系展示，取第0个国家
for i in range(40):
    plt.scatter(X[:,i],Y[:,0],color=randomcolor(),alpha=0.5,s=20)
    plt.ylim(bottom=0.00150, top=0.00325)
    plt.show()

# 计算加权后的相关性，绘图展示
import matplotlib.pyplot as plt
import seaborn as sns
# prevalence 和加权后ICA时间序列的关系展示
data = np.c_[Y1,X1]
data = pd.DataFrame(data)
corr=data.corr(method='pearson')
value = np.asanyarray(corr.iloc[1:,0])
plt.plot(value,'go:')
#sns.heatmap(data=corr)

#%% 检验和误差估计
from scipy.stats import f 
def Ftest(X,Y,Yhat,alpha):  
    n=len(X)  # 样本数
    k=X.shape[-1]  # 获取变量数
    f_arfa=f.isf(alpha, k, n-k-1)  # f临界值
        
    Yaver=Y.mean(axis=0) 
    U=((Yhat-Yaver)**2).sum(axis=0)
    Qe=((Y-Yhat)**2).sum(axis=0)
        
    F=(U/k)/(Qe/(n-k-1))
    answer=['F临界值:',f_arfa]
        
    if Y.ndim==1:
        answer.append(['函数F值:',F])
    else:
        for i in range(len(F)):
            answer.append(['函数'+str(i+1)+'的F值:',F[i]])
    
    return answer

def R(X,Y,Yhat):
    Yaver=Y.mean(axis=0) 
    
    fenzi=((Y-Yaver)*(Yhat-Yaver)).sum(axis=0)
    fenmu1=((Y-Yaver)**2).sum(axis=0)
    fenmu2=((Yhat-Yaver)**2).sum(axis=0)
    fenmu=np.sqrt(fenmu1*fenmu2)
    R=fenzi/fenmu
    return R
        
#%% 有权重的多元线性回归（拉直，只含原始项）,方法二
# 模型拟合
one = np.ones(len(X1))
X1 = np.c_[one,X1]
A1 = np.linalg.inv(X1.T@X1)@X1.T@Y1

Y1hat = X1@A1
err = (Y1-Y1hat)/Y1*100
Fvalue = Ftest(X1,Y1,Y1hat,0.05)
Rvalue = R(X1,Y1,Y1hat)
 
# 结果保存   
np.savetxt('d:\\heatwave and dementia\\data\\有权重的多元线性回归\\A(trend).csv',
           A1,delimiter = ',')
np.savetxt('d:\\heatwave and dementia\\data\\有权重的多元线性回归\\err(trend).csv',
           err,delimiter = ',')   

#%%  神经网络MLP回归（拉直，不含滞后项）

from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import KFold

#交叉验证
kf = KFold(n_splits=5)  # 5份交叉验证
xTrue=None
yTrue=None  
yHat=None  # 用于后续记录交叉验证合并的矩阵
#建立回归模型
mlp=MLPRegressor(hidden_layer_sizes=(20,), activation='logistic', solver='sgd', alpha=0.0001, 
                 learning_rate='adaptive', learning_rate_init=0.1, power_t=0.5, max_iter=5000, 
                 random_state=1, tol=0.0001, verbose=False, warm_start=False, momentum=0.9)
for trainIndex, testIndex in kf.split(X1):  # 建模、测试样本编号
    Xtrain, Xtest = X1[trainIndex], X1[testIndex]
    ytrain, ytest = Y1[trainIndex], Y1[testIndex]
    mlp.fit(Xtrain, ytrain)
    ypred = mlp.predict(Xtest)
    if yTrue is None:  # 第一份预测，当时yTrue是None
        xTrue = Xtest
        yTrue = ytest  # 真值
        yHat  = ypred  #预测值
    else:
        xTrue = np.r_[xTrue,Xtest]
        yTrue=np.r_[yTrue,ytest]  #  后续预测，进行行叠加
        yHat=np.r_[yHat, ypred]
        
err=np.sum(np.abs(yTrue-yHat)/yTrue*100)/len(X1) # 平均%误差
err=err.round(3)
Fvalue = Ftest(xTrue,yTrue,yHat,0.05)
Rvalue = R(xTrue,yTrue,yHat)

#保存模型
'''import pickle #序列化模块
with open('d:\\heatwave and dementia\\data\\有权重的多元线性回归\\mlpModel(trend).bin','wb') as f:
    rs = pickle.dumps(mlp)
    f.write(rs)
f.close()
np.savetxt('d:\\heatwave and dementia\\data\\有权重的多元线性回归\\mlperr(trend).csv',
           err,delimiter = ',')'''

#%% sklearn多项式回归
X_data = X1
Y_data = Y1

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

#交叉验证
kf = KFold(n_splits=5)  # 5份交叉验证
xTrue=None
yTrue=None  
yHat=None  # 用于后续记录交叉验证合并的矩阵
#建立回归模型
poly=PolynomialFeatures(degree=2)
regressor=LinearRegression()
for trainIndex, testIndex in kf.split(X1):  # 建模、测试样本编号
    Xtrain, Xtest = X1[trainIndex], X1[testIndex]
    ytrain, ytest = Y1[trainIndex], Y1[testIndex]
    poly_x_train=poly.fit_transform(Xtrain)
    regressor.fit(poly_x_train,ytrain)
    poly_x_test = poly.fit_transform(Xtest)
    ypred = regressor.predict(poly_x_test)
    if yTrue is None:  # 第一份预测，当时yTrue是None
        xTrue = poly_x_test
        yTrue = ytest  # 真值
        yHat  = ypred  #预测值
    else:
        xTrue = np.r_[xTrue,poly_x_test]
        yTrue=np.r_[yTrue,ytest]  #  后续预测，进行行叠加
        yHat=np.r_[yHat, ypred]
        
err=np.sum(np.abs(yTrue-yHat)/yTrue*100)/len(X1) # 平均%误差
err=err.round(3)
Fvalue = Ftest(xTrue,yTrue,yHat,0.05)
Rvalue = R(xTrue,yTrue,yHat)

#%% 随机森林回归

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from scipy.stats import spearmanr, pearsonr

#交叉验证
kf = KFold(n_splits=5)  # 5份交叉验证
xTrue=None
yTrue=None  
yHat=None  # 用于后续记录交叉验证合并的矩阵
#建立回归模型
rf = RandomForestRegressor(n_estimators=500, oob_score=True, random_state=0)
for trainIndex, testIndex in kf.split(X1):  # 建模、测试样本编号
    Xtrain, Xtest = X1[trainIndex], X1[testIndex]
    ytrain, ytest = Y1[trainIndex], Y1[testIndex]
    rf.fit(Xtrain, ytrain)
    ypred = rf.predict(Xtest)
    if yTrue is None:  # 第一份预测，当时yTrue是None
        yTrue = ytest  # 真值
        yHat  = ypred  #预测值
    else:
        yTrue=np.r_[yTrue,ytest]  #  后续预测，进行行叠加
        yHat=np.r_[yHat, ypred]

test_score = r2_score(yTrue, yHat)
spearman = spearmanr(yTrue, yHat)
pearson = pearsonr(yTrue, yHat)

print('Out-of-bag R-2 score estimate:', rf.oob_score_)
print('Test data R-2 score:', test_score)
print('Test data Spearman correlation:',spearman[0])
print('Test data Pearson correlation:',pearson[0])

#%% 支持向量机
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.svm import SVR

#交叉验证
kf = KFold(n_splits=5)  # 5份交叉验证
xTrue=None
yTrue=None  
yHat=None  # 用于后续记录交叉验证合并的矩阵
#建立回归模型
clf = SVR(C=1.0, cache_size=200, coef0=0.0, degree=2, epsilon=0.2, gamma='auto',
    kernel='rbf', max_iter=-1, shrinking=True, tol=0.001, verbose=False)
for trainIndex, testIndex in kf.split(X1):  # 建模、测试样本编号
    Xtrain, Xtest = X1[trainIndex], X1[testIndex]
    ytrain, ytest = Y1[trainIndex], Y1[testIndex]
    clf.fit(Xtrain, ytrain)
    ypred = clf.predict(Xtest)
    if yTrue is None:  # 第一份预测，当时yTrue是None
        xTrue = Xtest
        yTrue = ytest  # 真值
        yHat  = ypred  #预测值
    else:
        xTrue = np.r_[xTrue,Xtest]
        yTrue=np.r_[yTrue,ytest]  #  后续预测，进行行叠加
        yHat=np.r_[yHat, ypred]

err=np.sum(np.abs(yTrue-yHat)/yTrue*100)/len(X1) # 平均%误差
err=err.round(3)
Fvalue = Ftest(xTrue,yTrue,yHat,0.05)
Rvalue = R(xTrue,yTrue,yHat)

print(Fvalue)
print("R方:",Rvalue)
print("err:",err)




#################################################
#%% 有权重的多元线性回归（拉直，考虑滞后项,自回归项，ADL模型）

m = 161
n = 40
t = 30

#重构含滞后项的数据
def constructXY_lag(X,Y,q):
    for i in range(m):
        if i==0:
            Y2 = Y[q:30:,i]
        else:
            Y2 = np.r_[Y2,Y[q:30,i]]
    for i in range(q):
        for j in range(m):
            if j==0:
                xx = X[i:30-q+i,:]*W[j,:]
            else:
                xx = np.r_[xx,X[i:30-q+i]*W[j,:]]
        if i==0:
            X2 = xx
        else:
            X2 = np.c_[X2,xx]
    one = np.ones(len(X2))
    X2 = np.c_[one,X2]
    return X2,Y2

#重构含滞后项，递减分布的数据
def constructXY_lagdown(X,Y,q):
    for i in range(m):
        if i==0:
            Y22 = Y[q:30:,i]
        else:
            Y22 = np.r_[Y22,Y[q:30,i]]
    for i in range(m):
        r = 1.0
        for j in range(q):
            if j==0:
                xx = X[j:30-q+j]
            else:
                xx = xx+r*X[j:30-q+j]
            r = r/2
        if i==0:
            X22 = xx
        else:
            X22 = np.r_[X22,xx]
    one = np.ones(len(X22))
    X22 = np.c_[one,X22]
    return X22,Y22



#重构含自回归项的数据(未完成)
def constructXY_AR(X,Y,p):
    for i in range(m):
        if i==0:
            Y3 = Y[p+1:30,i]
        else:
            Y3 = np.r_[Y3,Y[p+1:30,i]]
    for i in range(p):
        for j in range(m):
            if j==0:
                yy = Y[p-i:30-1-i,j]
            else:
                yy = np.r_[yy,Y[p-i:30-1-i,j]]
        if i==0:
            X3 = yy
        else:
            X3 = np.c_[X3,yy]
    return X3,Y3




#%% 计算相关性确定滞后
    
X2,Y2 = constructXY_lag(X,Y,7)

import pandas as pd
import matplotlib.pyplot as plt

for i in range(40):
    data = Y2
    for j in range(7):
        data = np.c_[data,X2[:,j*40+i]]
    data = pd.DataFrame(data)
    corr=data.corr(method='pearson')
    value = np.asanyarray(corr.iloc[1:,0])
    print(np.shape(value))
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(value, 'go:')
    ax.set_title('ICA'+str(i), fontsize=16)
    ax.set_xlabel('lag', fontsize=14)
    ax.set_ylabel('corr', fontsize=14)
    
#%% 计算相关性确定自回归
    
X3,Y3 = constructXY_AR(X,Y,7)

import pandas as pd
import matplotlib.pyplot as plt

data = np.c_[Y3,X3]
data = pd.DataFrame(data)
corr=data.corr(method='pearson')
value = np.asanyarray(corr.iloc[1:,0])
print(np.shape(value))
fig, ax = plt.subplots(figsize=(14, 4))
ax.plot(value, 'go:')
ax.set_title('AR_corr', fontsize=16)
ax.set_xlabel('lag', fontsize=14)
ax.set_ylabel('corr', fontsize=14)    
    
#%% 构造独立滞后数据

X_data,Y_data = constructXY_lag(X,Y,7)
A = np.linalg.inv(X_data.T@X_data)@X_data.T@Y_data
Y_hat = X_data@A
err = (Y_data-Y_hat)/Y_data*100
Fvalue = Ftest(X_data,Y_data,Y_hat,0.05)
Rvalue = R(X_data,Y_data,Y_hat)
print("滞后期为：",6)
print(Fvalue)
print("R方：",Rvalue)
    
#%% 构造递减滞后数据
X_data,Y_data = constructXY_lagdown(X,Y,4)
A = np.linalg.inv(X_data.T@X_data)@X_data.T@Y_data
Y_hat = X_data@A
err = (Y_data-Y_hat)/Y_data*100
Fvalue = Ftest(X_data,Y_data,Y_hat,0.05)
Rvalue = R(X_data,Y_data,Y_hat)
print("滞后期为：",3)
print(Fvalue)
print("R方：",Rvalue)
        
#%% 求解模型
A2 = np.linalg.inv(X2.T@X2)@X2.T@Y2

#误差估计
Y2hat = X2@A2
err = (Y2-Y2hat)/Y2*100
Fvalue = Ftest(X2,Y2,Y2hat,0.05)
Rvalue = R(X2,Y2,Y2hat)

#保存结果
np.savetxt('d:\\heatwave and dementia\\data\\有权重的多元线性回归\\滞后递减A.csv',
           A2,delimiter = ',')
np.savetxt('d:\\heatwave and dementia\\data\\有权重的多元线性回归\\滞后递减err.csv',
           err,delimiter = ',')
#np.savetxt('d:\\heatwave and dementia\\data\\有权重的多元线性回归\\F.txt',
#           F)
#np.savetxt('d:\\heatwave and dementia\\data\\有权重的多元线性回归\\R.csv',
#           R,delimiter = ',')   

   
#%%  神经网络回归（含滞后项）

from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor

#分训练集和测试集
X_train, X_test, Y_train, Y_test = train_test_split(X2, Y2,test_size=0.05)

#建立回归模型
mlp=MLPRegressor(hidden_layer_sizes=(20,), activation='logistic', solver='sgd', alpha=0.0001, 
                 learning_rate='adaptive', learning_rate_init=0.1, power_t=0.5, max_iter=5000,  random_state=1, tol=0.0001, verbose=False, warm_start=False, momentum=0.9)

#模型训练和预测
mlp.fit(X_train,Y_train)
YPred=mlp.predict(X_test)
err=np.abs(Y_test-YPred)/Y_test*100
err=err.round(3)
F = Ftest(X_test,Y_test,YPred,0.05)
Rvalue = R(X_test,Y_test,YPred)

#保存模型
'''import pickle #序列化模块
with open('d:\\heatwave and dementia\\data\\有权重的多元线性回归\\mlpModel滞后.bin','wb') as f:
    rs = pickle.dumps(mlp)
    f.write(rs)
f.close()
np.savetxt('d:\\heatwave and dementia\\data\\有权重的多元线性回归\\mlperr滞后.csv',
           err,delimiter = ',')'''











