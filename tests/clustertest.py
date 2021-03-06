import scipy as sp
import numpy as np
import scipy.io
from importlib.machinery import SourceFileLoader

cluster = SourceFileLoader("cluster2D", "Unsupervised/cluster2D.py").load_module()

datas = sp.io.loadmat('./test_data/cluster/ex7data2.mat')
X = np.array(datas['X'])
#print(X)
inst = cluster.Cluster2D(4,X.T)
assert len(inst.cluster_centroids) == 4, 'error in creating centroid'
assert inst.cluster_centroids[0].shape == (2,1) , 'error in initializing cluster centroid' #testing centroid dimension

inst.Cluster(20)
