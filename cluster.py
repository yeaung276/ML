import numpy as np

class Cluster():
    def __init__(self,cluster,data):
        n,m = data.shape
        self.data = data
        self.cluster_no = cluster
        print(data[:,np.random.randint(0,m)].shape)
        self.cluster_centroids = [data[:,np.random.randint(0,m)].reshape(n,1) for i in range(cluster)]

    def get_distances(self):
        dists=[]
        for centroid in self.cluster_centroids:
            temp = self.data - centroid
            temp = np.square(temp)
            temp = np.sum(temp, axis = 0)
            dists.append(temp)
        self.__dists = np.array(dists)
        return np.array(dists)

    def get_centroids(self):
        n, m = self.data.shape
        self.__choosen_centroids = self.__dists.argmin(axis = 0).reshape(1,m)

    def __find_mean(self):
        for i in range(self.cluster_no):
            index = np.array((self.__choosen_centroids == i) * 1 )
            avg = np.dot(self.data, index.T)
            avg = avg/np.sum(index)
           
            self.cluster_centroids[i] = avg
        
    
    def update(self, iter , callback = lambda cent : None):
        for i in range(iter):
            self.get_distances()
            self.get_centroids()
            self.__find_mean()
            callback(self.cluster_centroids)
        return self.cluster_centroids

    def destortion_cost(self):
        pass

