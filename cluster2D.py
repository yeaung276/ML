import numpy as np
import matplotlib.pyplot as plt

import cluster as cluster
#TODO: implement K-mean algorithm
class Cluster2D(cluster.Cluster):
    def __init__(self, cluster, data):
        super().__init__(cluster, data)
    
    def plot_data(self):
        plt.style.use('seaborn-whitegrid')
        x = self.data[0,:]
        y = self.data[1,:]
        centroids = np.array(self.cluster_centroids)
        plt.scatter(x, y, marker = 'o', color = 'black', label= "data")
        plt.scatter(centroids[0,:], centroids[1,:], marker = 'o', color = 'red', label = "centroid")
        plt.legend(loc='upper right')
        plt.show()

