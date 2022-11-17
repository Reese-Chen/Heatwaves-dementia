#%% load data
import numpy as np

hw = np.loadtxt("d:\\heatwave and dementia\\data\\hw(第一列是编号).csv",
                delimiter=',',dtype=float)
hw = hw[:,1:]
######## remove seasonal trend with diff function
#%% define diff function 
'''def difference(dataset, interval=1):
	diff = list()
	for i in range(interval, len(dataset)):
		value = dataset[i] - dataset[i - interval]
		diff.append(value)
	return diff

#%% remove seasonal trend with diff function
    
x = hw[:,0]
dx = difference(x,4)

fig, axes = plt.subplots(2, 1)
axes[0].plot(x[:1000])
axes[0].title.set_text('data')

axes[1].plot(dx[:1000])
axes[1].title.set_text('diff')

#%%
x = None
for i in range(19862):
    dx = difference(hw[:,i],365)
    if i==0:
        x = dx
    else:
        x = np.c_[x,dx]
#%% write to csv
x = x[:,4406:]
np.savetxt('d:\\heatwave and dementia\\data\\hw_without_trend.csv',x, delimiter = ',')

#%% visilization
data = np.average(hw,axis=1)
diff = np.average(x,axis=1)
fig, axes = plt.subplots(2, 1)
axes[0].plot(data)
axes[0].title.set_text('data')

axes[1].plot(diff)
axes[1].title.set_text('diff')


######## remove seasonal trend with poly function
#%% show the original trend

# fit polynomial: x^2*b1 + x*b2 + ... + bn
X = [i%365 for i in range(0, 10957)]
y = hw[:,0]
degree = 4
coef = np.polyfit(X, y, degree)
print('Coefficients: %s' % coef)
# create curve
curve = list()
for i in range(len(X)):
	value = coef[-1]
	for d in range(degree):
		value += X[i]**(degree-d) * coef[d]
	curve.append(value)
# plot curve over original data
plt.plot(y,linewidth=0.2)
plt.plot(curve, color='red', linewidth=1)
plt.show()

# fit polynomial: x^2*b1 + x*b2 + ... + bn
diff = list()
for i in range(10957):
	value = y[i] - curve[i]
	diff.append(value)
plt.plot(diff,linewidth=0.2)
plt.show()

#%% remove the trend

# fit polynomial and get average coeficients
X =  [i%365 for i in range(0, 10957)]
degree = 4
for i in range(15457):
    y = hw[:,i]
    coef = np.polyfit(X,y,degree)
    if i==0:
        meancoef = coef
    else:
        meancoef = meancoef+coef
meancoef = meancoef/15457

#%%
# create curve
curve = list()
for i in range(len(X)):
	value = meancoef[-1]
	for d in range(degree):
		value += X[i]**(degree-d) * meancoef[d]
	curve.append(value)
    
#%%   
# minus curve
newhw = None
for i in range(15457):
    value = hw[:,i]-curve
    if i==0:
        newhw = value
    else:
        newhw = np.c_[newhw,value]
#%%
np.savetxt('d:\\heatwave and dementia\\data\\hw_without_trend1.csv',newhw,delimiter = ',')
#%% visializatioin

diff1 = np.average(newhw,axis=1)
plt.plot(diff,linewidth=0.2)
plt.show()'''


#%%
X =  [i%365 for i in range(0, 10957)]

degree = 4
for i in range(15456):
    y = hw[:,i]
    coef = np.polyfit(X,y,degree)
    curve = np.polyval(coef,X)
    value = y-curve
    if i==0:
        newhw = value
    else:
        newhw = np.c_[newhw,value]
#np.savetxt('d:\\heatwave and dementia\\data\\hw_without_trend.csv',newhw,delimiter = ',')

#%%
        
import matplotlib.pyplot as plt

data0 = np.average(hw,axis=1)        
plt.plot(data0,linewidth=0.3)
#plt.label("original series")
plt.show()
data1 = np.average(newhw,axis=1)       
plt.plot(data1,linewidth=0.3)      
#plt.label("series without trend")       
plt.show()    

#%%       
plt.plot(data0[:1000],linewidth=0.3)      
plt.plot(data1[:1000],linewidth=0.3)      