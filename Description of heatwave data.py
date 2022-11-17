import numpy as np
import pandas as pd

#%% combine heatwave data with countryname
hw = np.loadtxt("d:\\heatwave and dementia\\data\\heatwave.csv",
                delimiter=',',dtype=float)
cn = np.loadtxt("d:\\heatwave and dementia\\data\\hw_english_countrynames.csv",
                delimiter=',',encoding='utf_8',dtype=str)

hw = pd.DataFrame(hw)
hw = hw.iloc[:,4406:]
hw.columns = cn

#%% reform the data

hw1 = hw
hw1.loc['Col_sum'] = hw1.apply(lambda x: x.sum())

hw2 = hw1.iloc[[10957]]
countryname = hw2.columns
val = np.array(hw2)

hw3 = np.c_[countryname.T,val.T]
hw3 = pd.DataFrame(hw3)
hw3.columns = ["countryname","val"]
hw3[['val']] = hw3[['val']].astype(float)

#%% describe the characteristics of heatwaves for each country

hw4 = hw3.groupby("countryname")["val"].mean()
hw5 = hw3.groupby("countryname").count()
hw4.to_csv('d:\\heatwave and dementia\\data\\hw4.csv', encoding='utf_8')
hw5.to_csv('d:\\heatwave and dementia\\data\\hw5.csv', encoding='utf_8')

#%% visiliaze the distribution of heatwaves

from pyecharts import options as opts
from pyecharts.charts import Map
from pyecharts.faker import Faker

hw4 = hw3.groupby("countryname")["val"].mean()
print(hw4.max(),' ',hw4.min())
xarea = list(hw4.index)
ynum = list(hw4)

for i,name in enumerate(xarea):
    if name == 'Congo':
        xarea[i] = 'Dem.Rep.Congo'
    if name == 'Central African Republic':
        xarea[i] = 'Central African Rep.'
    if name == 'Sudan':
        xarea[i] = 'S.Sudan'
    if name == 'Guinea':
        xarea[i] = 'Eq.Guinea'
    if name == 'Libyan Arab Jamahiriya':
        xarea[i] = 'Libya'
    if name == 'Syrian Arab Republic':
        xarea[i] = 'Syria'
    if name == "Lao People's Democratic Republic":
        xarea[i] = 'Lao PDR'
    if name == 'Viet Nam':
        xarea[i] = 'Vietnam'
    if name == 'Sudan':
        xarea[i] = 'S.Sudan'
        
c = (
    Map()
    .add("sum of HWMId", [list(z) for z in zip(xarea, ynum)], "world")
    .set_series_opts(label_opts=opts.LabelOpts(is_show=False))
    .set_global_opts(
        title_opts=opts.TitleOpts(title="Heatwave distribution"),
        visualmap_opts=opts.VisualMapOpts(min_=6092,max_=7478,is_piecewise=True),
    )
    .render("d:\\heatwave and dementia\\data\\heatwaves_visialization.html")
)

#%% visiliaze the trend of heatwaves

import matplotlib.pyplot as plt

hw1['Row_sum'] = hw1.apply(lambda x: x.sum(), axis=1)
#%%
hw1.drop(index = 'Col_sum',inplace = True)
hw1.Row_sum.plot()
#%%

hw6 = list(hw1.Row_sum)
dt1 = pd.date_range(start="19900101", end="20191231", freq="D")

plt.plot(dt1,hw6,color = 'teal',linestyle='-',linewidth=0.2)
plt.xlabel('time', fontsize=15) # x轴名称
plt.ylabel('heatwave', fontsize=15) # y轴名称

from sklearn.linear_model import LinearRegression
dt1 = np.array(dt1).reshape(-1, 1)
hw6 = np.array(hw6).reshape(-1, 1)
reg = LinearRegression()
reg.fit(dt1,hw6)
a = reg.coef_[0][0]     # 系数
b = reg.intercept_[0]   # 截距
dt2 = np.arange(10957)
prediction = a*dt2+b
#plt.plot(dt2,prediction,color='darkslategrey',
#         linestyle='-',label='Fit curve',linewidth=1.0)

plt.show()


