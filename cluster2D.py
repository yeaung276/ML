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
        plt.scatter(x, y, marker = '.', color = 'black', label= "data")
        plt.scatter(centroids[:,0,:], centroids[:,1,:], marker = 'o', color = 'red', label = "centroid")
        plt.legend(loc='upper right')
        plt.show()

    def Cluster(self,iter):
        colors = ['blue', 'green', 'yellow', 'pink', 'orange']
        K = self.cluster_no
        self.t = [[] for i in range(K)]
        self.update(iter,self.track)
        plt.style.use('seaborn-whitegrid')
        x = self.data[0,:]
        y = self.data[1,:]
        centroids = np.array(self.t)
        plt.scatter(x, y, marker = '.', color = 'black', label= "data")
        for i in range(self.cluster_no):
            plt.plot(centroids[i,:,0,:], centroids[i,:,1,:], marker = 'x', color = colors[i], label = "centroid{}".format(i))
        c = np.array(self.cluster_centroids)
        plt.scatter(c[:,0,:], c[:,1,:], marker = 'o', color = 'red', label = "centroid")
        plt.legend(loc='upper right')
        plt.show()

    def track(self,centroids):
        K = self.cluster_no
        for i in range(K):
            self.t[i].append(centroids[i])