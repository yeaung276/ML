import numpy as np

class Data():
   
    def __init__(self, X, size, label = np.array([[]])):
        n,m = X.shape
        if(m == size):
            self.data = X
            self.m = m
            self.mu = self.__find_mean()
            self.sq_sigma = self.__find_varience()
            self.sigma = np.sqrt(self.sq_sigma)
            self.Y = self.set_label(label)
        else:
            print("size and matrix size doesn't match")

    def __find_mean(self):
        m = self.m
        return (1/m) * np.sum(self.data, axis=1, keepdims=True)

    def __find_varience(self):
        m = self.m
        varience = (1/m) * np.sum(np.square(self.data - self.mu), axis=1, keepdims=True)
        return varience

    def info(self):
        information = {}
        print('Data set size: {}'.format(self.m),end='\n')
        information['data_size'] = self.m
        print('Feature Dimension: {}'.format(self.data.shape[0]), end='\n')
        information['feature_size'] = self.data.shape[0]
        print('labeled data:  {}'.format((self.Y != 0).all()))
        information['labeled_data'] = (self.Y != 0).all()

    def Normalize(self):
        m = self.m
        N_data = (self.data - self.mu)/self.sigma
        return Data(N_data,m)

    def featureScale(self , Scaletype = 'min_max'):
        functions = { 
        'min_max': self.min_max,
        'max_norm': self.max_norm
        }
        functions.get(Scaletype,lambda : 'not found')()
            

    def min_max(self):
        m = self.m
        max_values = np.max(self.data, axis=1, keepdims=True)
        min_values = np.min(self.data, axis=1, keepdims=True)
        values = (self.data - min_values)/(max_values - min_values)
        self.data = values
        return self
        
    def max_norm(self):
        m = self.m
        max_values = np.max(self.data, axis=1, keepdims=True)
        values = self.data/max_values
        self.data = values
        return self
    
    def mini_batch(self, batchsize = 1):
        m = self.m
        complete_batches = int(m/batchsize)
        remainder = m%batchsize
        batches = []
        start = 0
        for i in range(1,complete_batches+1):
            batches.append(Data(self.data[:,start:i*batchsize],batchsize, label = self.Y[:,start:i*batchsize]))
            start = i*batchsize
        
        batches.append(Data(self.data[:,start:m],remainder, label = self.Y[:,start:m]))
        return batches
    
    def set_label(self,Y):
        n,m = Y.shape
        if(m == 0):
            return np.zeros((1,self.m))
        if(m != self.m):
            print("size and matrix size doesn't match")
            return np.zeros((1,self.m))
        else:
            return Y
        
    def one_hot_encode(self,depth):
        """labeled class start with 1 ,not 0"""
        m = self.m
        encode = np.zeros((depth,m))
        for i,n in self.Y[0,:]:
            encode[n-1,i] = 1
        return encode

    def shuffle(self):
        m = self.m
        permutation = np.arange(m)
        np.random.shuffle(permutation)
        self.data = self.data[:,permutation]
        self.Y = self.Y[:,permutation]
        return self


    