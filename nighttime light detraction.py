#%% build connection
import ee
import os
ee.Authenticate()
if __name__ == '__main__':
    os.environ['HTTP_PROXY'] = 'http://127.0.0.1:10809'
    os.environ['HTTPS_PROXY'] = 'https://127.0.0.1:10809'
ee.Initialize()

#%% load data
nighttime = ee.ImageCollection('NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG').select('avg_rad').filter(ee.Filter.date('2014-01-01', '2019-12-31'))

#%% get data at an exact point
def getdata(lon,lat): 
    poi = ee.Geometry.Point(lon, lat)
    scale = 1000
    return nighttime.getRegion(poi, scale).getInfo()

#%% transform data to time series
import pandas as pd
def ee_array_to_df(arr, list_of_bands):
    """Transforms client-side ee.Image.getRegion array to pandas.DataFrame."""
    df = pd.DataFrame(arr)

    # Rearrange the header.
    headers = df.iloc[0]
    df = pd.DataFrame(df.values[1:], columns=headers)

    # Remove rows without data inside.
    df = df[['longitude', 'latitude', 'time', *list_of_bands]].dropna()

    # Convert the data to numeric values.
    for band in list_of_bands:
        df[band] = pd.to_numeric(df[band], errors='coerce')

    # Convert the time field into a datetime.
    df['datetime'] = pd.to_datetime(df['time'], unit='ms')

    # Keep the columns of interest.
    #df = df[['longitude', 'latitude','time','datetime',  *list_of_bands]]
    df = df[['longitude', 'latitude','datetime','time', *list_of_bands]]

    return df

#%% detract all points on land
import numpy as np
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
                
#%% detract nightlight data on each points
nightlight = None
for i in range(19285,19862):
    lon = int(location[0,i])
    lat = int(location[1,i])
    nighttime_poi = getdata(lon,lat)
    nighttime_val = ee_array_to_df(nighttime_poi,['avg_rad'])
    if nightlight is None:
        nightlight = list(nighttime_val.avg_rad)
    else:
        nightlight = np.c_[nightlight,list(nighttime_val.avg_rad)]
                
#%% save the data to csv
np.savetxt('夜间灯光1.csv', nightlight, delimiter = ',')             

#%% visialization changes
import matplotlib.pyplot as plt
from scipy import optimize

lon = 4.8148
lat = 45.7758
nighttime_poi = getdata(lon,lat)
nighttime_val = ee_array_to_df(nighttime_poi,['avg_rad'])

# Fitting curves.
## First, extract x values (times) from the dfs.
x_data = np.asanyarray(nighttime_val['time'].apply(float))

## Secondly, extract y values (LST) from the dfs.
#y_data = np.asanyarray(nighttime_val['avg_rad'].apply(float))
y_data = nightlight.sum(axis=1)

# Subplots.
fig, ax = plt.subplots(figsize=(14, 6))

# Add scatter plots.
ax.scatter(nighttime_val['datetime'], y_data,
           c='green', alpha=0.2, label='nighttime light')

# Add fitting curves.
z1 = np.polyfit(x_data,y_data,4)
y_fit = np.polyval(z1,x_data) 
ax.plot(nighttime_val['datetime'],y_fit,label='fitted', color='green', lw=2.5)

# Add some parameters.
ax.set_title('Nighttime Light', fontsize=16)
ax.set_xlabel('Date', fontsize=14)
ax.set_ylabel('avg_rad', fontsize=14)
ax.legend(fontsize=14, loc='lower right')

plt.show()

#%% visialization distribution
# Reduce the LST collection by mean.
nighttime_img = nighttime.mean()
nighttime_img = nighttime_img.select('avg_rad')

from IPython.display import Image

# Create a URL to the styled image for a region around France.
url = nighttime_img.getThumbUrl({
    'min': 0, 'max': 60, 'dimensions': 512,
    'palette': ['black', 'white', 'orange', 'red']})
print(url)

# Display the thumbnail land surface temperature in France.
Image(url=url)
