import numpy as np
import pandas as pd

from pyecharts import options as opts
from pyecharts.charts import Map
from pyecharts.faker import Faker

import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression

#%% read the data
incidence = pd.read_csv(r'd:\\heatwave and dementia\\data\\incidence of dementia.csv',
                        sep=',',header='infer')
prevalence = pd.read_csv(r'd:\\heatwave and dementia\\data\\prevalence of dementia.csv',
                        sep=',',header='infer')

incidence = pd.DataFrame(incidence)
prevalence = pd.DataFrame(prevalence)


######incidence
#%% delete unused data

countryname = pd.read_csv('d:\\heatwave and dementia\\data\\痴呆国家列表1.csv',
                         sep=',',header=None)
#countryname = np.loadtxt('d:\\heatwave and dementia\\data\\痴呆国家列表1.csv',
#                         delimiter=',',dtype=str)
countryname = countryname.iloc[:,0]
countryname = np.array(countryname)
incidence = incidence[incidence["location_name"].isin(countryname)]

#%% rename the countries

pd.set_option('mode.chained_assignment', None)
incidence.location_name[incidence.location_name=="Bolivia (Plurinational State of)"]="Bolivia"
incidence.location_name[incidence.location_name=="Côte d'Ivoire"]="C?te d'Ivoire"
incidence.location_name[incidence.location_name=="Czechia"]="Czech Republic"
#incidence.location_name[incidence.location_name=="Dominica"]="Dominican Republic"
incidence.location_name[incidence.location_name=="Iran (Islamic Republic of)"]="Iran"
incidence.location_name[incidence.location_name=="Libya"]="Libyan Arab Jamahiriya"
incidence.location_name[incidence.location_name=="Palestine"]="Palestinian Territory"
incidence.location_name[incidence.location_name=="Republic of Korea"]="Korea"
incidence.location_name[incidence.location_name=="Russian Federation"]="Russia"
incidence.location_name[incidence.location_name=="Solomon Islands"]="Solomon Islands (the)"
incidence.location_name[incidence.location_name=="Taiwan (Province of China)"]="Taiwan"
incidence.location_name[incidence.location_name=="United Republic of Tanzania"]="Tanzania"
incidence.location_name[incidence.location_name=="United States of America"]="United States"
incidence.location_name[incidence.location_name=="Venezuela (Bolivarian Republic of)"]="Venezuela"

#%% save the new data

incidence.to_csv('d:\\heatwave and dementia\\data\\incidence of dementia(1).csv', encoding='utf_8')

#%% visiliaze the distribution of dementia

incidence1 = incidence.where((incidence.sex_id==3)&(incidence.age_id==22)).dropna()
countryname = incidence1.location_name
val = incidence1.val
val = np.float64(val)
data = np.c_[countryname,val]
data = pd.DataFrame(data)
data.columns = ["countryname","val"]
data[['val']] = data[['val']].astype(float)
data1 = data.groupby("countryname")["val"].mean()
print(data1.min(),' ',data1.max())
xarea = list(data1.index)
ynum = list(data1)

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
    .add("mean value of incidence of dementia", [list(z) for z in zip(xarea, ynum)], "world")
    .set_series_opts(label_opts=opts.LabelOpts(is_show=False))
    .set_global_opts(
        title_opts=opts.TitleOpts(title="Incidence distribution"),
        visualmap_opts=opts.VisualMapOpts(min_=1.83e-05,max_=0.00051,is_piecewise=True),
    )
    .render("d:\\heatwave and dementia\\data\\incidence_visialization.html")
)

#%%describe the characteristics of heatwaves for each country

data1.to_csv('d:\\heatwave and dementia\\data\\incidence平均.csv', encoding='utf_8')

#%% visiliaze the trend of incidence of dementia

incidence2 = incidence.where((incidence.sex_id==1)&(incidence.age_id==22)).dropna()
incidence3 = incidence.where((incidence.sex_id==2)&(incidence.age_id==22)).dropna()
Both = incidence1.groupby("year")["val"].mean()
Male = incidence2.groupby("year")["val"].mean()
Female = incidence3.groupby("year")["val"].mean()

dt1 = pd.date_range(start="19900101", end="20191231", freq="Y")

plt.plot(dt1,Both,color = 'darkslategray',linestyle='dashed',
         label='Both sex',linewidth=2.0)
plt.plot(dt1,Male,color = 'orange',linestyle='-',
         label='Male',linewidth=2.0)
plt.plot(dt1,Female,color = 'teal',linestyle='-',
         label='Female',linewidth=2.0)
plt.xlabel('year', fontsize=14)
plt.ylabel('incidence of dementia', fontsize=14)
plt.legend(fontsize=14, markerscale=2., scatterpoints=1)
plt.show()
#%%
incidence4 = incidence.where((incidence.sex_id==3)&(incidence.age_id==232)).dropna()
incidence5 = incidence.where((incidence.sex_id==3)&(incidence.age_id==243)).dropna()
incidence6 = incidence.where((incidence.sex_id==3)&(incidence.age_id==160)).dropna()

allage = incidence1.groupby("year")["val"].mean()
sep1 = incidence4.groupby("year")["val"].mean()
sep2 = incidence5.groupby("year")["val"].mean()
sep3 = incidence6.groupby("year")["val"].mean()

dt1 = pd.date_range(start="19900101", end="20191231", freq="Y")

plt.plot(dt1,allage,color = 'darkslategray',linestyle='dashed',
         label='All ages',linewidth=2.0)
plt.plot(dt1,sep1,color = 'orange',linestyle='-',
         label='65-74',linewidth=2.0)
plt.plot(dt1,sep2,color = 'firebrick',linestyle='-',
         label='75-84',linewidth=2.0)
plt.plot(dt1,sep3,color = 'teal',linestyle='-',
         label='85 plus',linewidth=2.0)
plt.xlabel('year', fontsize=14)
plt.ylabel('incidence of dementia', fontsize=14)
plt.legend(fontsize=14, markerscale=2., scatterpoints=1)
plt.show()

######prevalence
#%% delete unused data

prevalence = prevalence[prevalence["location_name"].isin(countryname)]

#%% rename the countries

pd.set_option('mode.chained_assignment', None)
prevalence.location_name[prevalence.location_name=="Bolivia (Plurinational State of)"]="Bolivia"
prevalence.location_name[prevalence.location_name=="Côte d'Ivoire"]="C?te d'Ivoire"
prevalence.location_name[prevalence.location_name=="Czechia"]="Czech Republic"
#prevalence.location_name[prevalence.location_name=="Dominica"]="Dominican Republic"
prevalence.location_name[prevalence.location_name=="Iran (Islamic Republic of)"]="Iran"
prevalence.location_name[prevalence.location_name=="Libya"]="Libyan Arab Jamahiriya"
prevalence.location_name[prevalence.location_name=="Palestine"]="Palestinian Territory"
prevalence.location_name[prevalence.location_name=="Republic of Korea"]="Korea"
prevalence.location_name[prevalence.location_name=="Russian Federation"]="Russia"
prevalence.location_name[prevalence.location_name=="Solomon Islands"]="Solomon Islands (the)"
prevalence.location_name[prevalence.location_name=="Taiwan (Province of China)"]="Taiwan"
prevalence.location_name[prevalence.location_name=="United Republic of Tanzania"]="Tanzania"
prevalence.location_name[prevalence.location_name=="United States of America"]="United States"
prevalence.location_name[prevalence.location_name=="Venezuela (Bolivarian Republic of)"]="Venezuela"

#%% save the new data

prevalence.to_csv('d:\\heatwave and dementia\\data\\prevalence of dementia(1).csv', encoding='utf_8')

#%% visiliaze the distribution of dementia

prevalence1 = prevalence.where((prevalence.sex_id==3)&(prevalence.age_id==22)).dropna()
countryname = prevalence1.location_name
val = prevalence1.val
val = np.float64(val)
data = np.c_[countryname,val]
data = pd.DataFrame(data)
data.columns = ["countryname","val"]
data[['val']] = data[['val']].astype(float)
data1 = data.groupby("countryname")["val"].mean()
print(data1.min(),' ',data1.max())
xarea = list(data1.index)
ynum = list(data1)

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
    .add("mean value of prevalence of dementia", [list(z) for z in zip(xarea, ynum)], "world")
    .set_series_opts(label_opts=opts.LabelOpts(is_show=False))
    .set_global_opts(
        title_opts=opts.TitleOpts(title="Prevalence distribution"),
        visualmap_opts=opts.VisualMapOpts(min_=0.00056,max_=0.0179,is_piecewise=True),
    )
    .render("d:\\heatwave and dementia\\data\\prevalence_visialization.html")
)

#%%describe the characteristics of heatwaves for each country

data1.to_csv('d:\\heatwave and dementia\\data\\prevalence平均.csv', encoding='utf_8')

#%% visiliaze the trend of prevalence of dementia

prevalence2 = prevalence.where((prevalence.sex_id==1)&(prevalence.age_id==22)).dropna()
prevalence3 = prevalence.where((prevalence.sex_id==2)&(prevalence.age_id==22)).dropna()
Both = prevalence1.groupby("year")["val"].mean()
Male = prevalence2.groupby("year")["val"].mean()
Female = prevalence3.groupby("year")["val"].mean()

dt1 = pd.date_range(start="19900101", end="20191231", freq="Y")

plt.plot(dt1,Both,color = 'darkslategray',linestyle='dashed',
         label='Both sex',linewidth=2.0)
plt.plot(dt1,Male,color = 'orange',linestyle='-',
         label='Male',linewidth=2.0)
plt.plot(dt1,Female,color = 'teal',linestyle='-',
         label='Female',linewidth=2.0)
plt.xlabel('year', fontsize=14)
plt.ylabel('prevalence of dementia', fontsize=14)
plt.legend(fontsize=14, markerscale=2., scatterpoints=1)
plt.show()
#%%
prevalence4 = prevalence.where((prevalence.sex_id==3)&(prevalence.age_id==232)).dropna()
prevalence5 = prevalence.where((prevalence.sex_id==3)&(prevalence.age_id==243)).dropna()
prevalence6 = prevalence.where((prevalence.sex_id==3)&(prevalence.age_id==160)).dropna()

allage = prevalence1.groupby("year")["val"].mean()
sep1 = prevalence4.groupby("year")["val"].mean()
sep2 = prevalence5.groupby("year")["val"].mean()
sep3 = prevalence6.groupby("year")["val"].mean()

dt1 = pd.date_range(start="19900101", end="20191231", freq="Y")

plt.plot(dt1,allage,color = 'darkslategray',linestyle='dashed',
         label='All ages',linewidth=2.0)
plt.plot(dt1,sep1,color = 'orange',linestyle='-',
         label='65-74',linewidth=2.0)
plt.plot(dt1,sep2,color = 'firebrick',linestyle='-',
         label='75-84',linewidth=2.0)
plt.plot(dt1,sep3,color = 'teal',linestyle='-',
         label='85 plus',linewidth=2.0)
plt.xlabel('year', fontsize=14)
plt.ylabel('prevalence of dementia', fontsize=14)
plt.legend(fontsize=14, markerscale=2., scatterpoints=1)
plt.show()