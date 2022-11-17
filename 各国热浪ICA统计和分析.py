import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#%% load data
hw_ica = np.loadtxt('d:\\heatwave and dementia\\code\\ICAcomponents.csv',
                    delimiter = ',')
hw_cn = np.loadtxt("d:\\heatwave and dementia\\data\\hw_english_countrynames.csv",
                delimiter=',',encoding='utf_8',dtype=str)

hw_ica = hw_ica.T
hw_ica = pd.DataFrame(hw_ica)
hw_ica.insert(hw_ica.shape[1],'countryname',hw_cn)

#%% Filter the corresponding countries
countryname = pd.read_csv('d:\\heatwave and dementia\\data\\热浪国家列表1.csv',
                         sep=',',header=None)
countryname = countryname.iloc[:,0]
countryname = np.array(countryname)
hw_ica = hw_ica[hw_ica["countryname"].isin(countryname)]

#%% sum by country and apply nomorlization
hw_ica1 = hw_ica.groupby("countryname").sum()
#%% apply nomorlization
for i in range(163):
    a = hw_ica1.iloc[i]
    mean = np.mean(a)
    std = np.std(a)
    if(std):
        hw_ica1.iloc[i] = hw_ica1.iloc[i].apply(lambda x: (x - mean) / std)
        
#%% save to csv
hw_ica1.to_csv('d:\\heatwave and dementia\\data\\各国ICA.csv',sep=',',index=True,header=True)

#%% ICA anaylsis and visialization
total = np.zeros(40)

plt.rcParams["font.sans-serif"]=['SimHei']
plt.rcParams["axes.unicode_minus"]=False

for i in range(40):
    total[i] = np.sum(hw_ica1[i])
    plt.bar(i,total[i])

plt.title("Total share of ICAs")
plt.xlabel("ICA number")
plt.ylabel("Total share")
    
plt.show()

import seaborn as sns
plt.figure(dpi=120)
sns.heatmap(data=hw_ica1,vmin=0, vmax=3.29)

#%% transform to 0-1matrix
hw_ica2 = hw_ica1
for i in range(163):
    p90 = hw_ica2[i].quantile(0.9)
    hw_ica2[i] = hw_ica2[i]>p90
#%%
hw_ica2.to_csv('d:\\heatwave and dementia\\data\\各国ICA01.csv',sep=',',index=True,header=True)

#%% load 
mix = np.loadtxt('d:\\heatwave and dementia\\code\\ICAmix.csv',
                    delimiter = ',')
hw = np.loadtxt('d:\\heatwave and dementia\\data\\hw_without_trend.csv',delimiter = ',')
#%% caculate the time series of ICAS
'''ica_hw = np.zeros(10957,40)
for i in range(40):
    for j in range(15456):
        ica_hw[:,i] = ica_hw[:,i]+mix[j,i]*hw[:,j]'''
        
ica_hw = np.dot(hw,mix)
np.savetxt('d:\\heatwave and dementia\\data\\ica_hw.csv',ica_hw,delimiter = ',')












