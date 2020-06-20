import cluster2D as cluster
import scipy as sp
import numpy as np
import scipy.io

datas = sp.io.loadmat('./test_data/cluster/ex7data2.mat')
X = np.array(datas['X'])
#print(X)
inst = cluster.Cluster2D(3,X.T)
assert len(inst.cluster_centroids) == 3, 'error in creating centroid'
assert inst.cluster_centroids[0].shape == (2,1) , 'error in initializing cluster centroid' #testing centroid dimension

inst.Cluster(20)
