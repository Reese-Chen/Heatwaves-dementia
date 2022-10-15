

import numpy as np
import matplotlib.pyplot as plt
import numpy.random as npr
import csv

from sklearn import datasets,decomposition

def load_data():
    data = np.loadtxt(open("X.csv","rb"),delimiter=",") 
    return data

#超大规模数据集降维IncrementalPCA模型
def test_IncrementalPCA(data):
    X = data
    # 使用默认的 n_components
    pca=decomposition.IncrementalPCA(n_components=100,batch_size=10)
    pca.partial_fit(X)
    aa = pca.transform(X) #混合矩阵
    X1 = pca.components_  #降维后的主成分
    print('explained variance ratio : %s'% str(pca.explained_variance_ratio_))
    print(pca.n_components_)
    np.savetxt("X1.txt", X1)
    np.savetxt("M1.txt", aa)
    
# 产生用于降维的数据集
#X=load_data()
X = npr.randint(1,10,(32400,10950))
print(X[1:3,2:4])
# 调用 test_IncrementalPCA
test_IncrementalPCA(X)
