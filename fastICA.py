from sklearn.decomposition import FastICA
import numpy as np
import pandas as pd
from sica.base import MSTD

#%% load data
data = np.loadtxt("d:\\heatwave and dementia\\data\\W365_seperate.csv",
                delimiter=',',dtype=float)
data = data.T

#%% use MSTD to determine the number of ICAs

MSTD(data , m = 5 , M = 100 , step = 2 , n_runs = 50)


#%% run fastICA 

ica = FastICA(n_components=40,random_state=0,whiten='unit-variance')
X_transformed = ica.fit_transform(data)
print(X_transformed.shape)
print(ica.n_iter_)

#%% visialize the ICAs
import matplotlib.pyplot as plt
for i in range(40):
    plt.plot(X_transformed[:,i])
    plt.title("ICA"+str(i))
    plt.show()
    
#%% save ICA
np.savetxt('d:\\heatwave and dementia\\code\\ICAs.csv',X_transformed, delimiter = ',')

#%% save relevant matrix
components = ica.components_
mix = ica.mixing_
white = ica.whitening_
mean_feature = ica.mean_

np.savetxt('d:\\heatwave and dementia\\code\\ICAcomponents.csv',components, delimiter = ',')
np.savetxt('d:\\heatwave and dementia\\code\\ICAmix.csv',mix, delimiter = ',')
np.savetxt('d:\\heatwave and dementia\\code\\ICAwhitening.csv',white, delimiter = ',')
