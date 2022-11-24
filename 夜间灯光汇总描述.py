import numpy as np
from pyecharts import options as opts
from pyecharts.charts import Geo
from pyecharts.faker import Faker
import pandas as pd

#%% load data
nighttime = np.loadtxt(r"d:\\heatwave and dementia\\data\\夜间灯光1.csv",
                    delimiter=',')

#%% get mean value for each points

mean_night = nighttime.mean(axis=0)
sum_time_series = nighttime.sum(axis=1)

#%% generate coordinates for points
from global_land_mask import globe
k = 0
for i in range(-85,86,1):
    for j in range(-179,180,1):
        is_on_land = globe.is_land(i,j)
        if (is_on_land):
            k = k+1
            new = [j,i]
            if k==1:
                location = new
            else:
                location = np.column_stack((location,new))
location = location[:,4406:19285]          

#%% combine coordinate points and catelog
location = np.transpose(location)
indx = np.arange(14879)
data= np.c_[indx,location,mean_night]
data = pd.DataFrame(data,columns=['indx', 'lon', 'lat', 'catelog'])
 
#%% draw the picture
geo_sight_coord={data['indx'][i]: [data['lon'][i], data['lat'][i]] for i in range(len(data))} #构造位置字典数据
data_pair=[(data['indx'][i], data['catelog'][i]) for i in range(len(data))] #构造项目租金数据

g=Geo() #地理初始化
g.add_schema(maptype="world") #限定范围
for key, value in geo_sight_coord.items(): #对地理点循环
    g.add_coordinate(key, value[0], value[1]) #追加点位置
g.add("", data_pair, symbol_size=2) 
g.set_series_opts(label_opts=opts.LabelOpts(is_show=False), type='scatter')  #星散点图scatter

g.set_global_opts(
        visualmap_opts=opts.VisualMapOpts(type_="color",max_=0.04,min_=1), 
        title_opts=opts.TitleOpts(title="nighttime_light")
    )
g.render("d:\\heatwave and dementia\\data\\nighttime_light.html")



