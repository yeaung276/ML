import numpy as np

class neural_net():
    
    def __init__(self,config):
        self.__no_layers = len(config['architecture'])
        self.__training_data = 0
        self.__activations= [ np.zeros((n,1)) for n in config['architecture'] ]
        self.__dalta=[ np.zeros((n,1)) for n in config['architecture'] ]
        if 'rand_range' in config:
            self.__weights = [ (np.random.rand(config['architecture'][n+1],config['architecture'][n]+1)*2*config['rand_range'])-config['rand_range'] for n in range(self.__no_layers-1) ]
        else:
            self.__weights = [ np.random.rand(config['architecture'][n+1],config['architecture'][n]+1) for n in range(self.__no_layers-1) ]
    
    def __sigmoid(self,x):
        #sigmoid for calculting the activation value
        return 1/(1+np.exp(-x))

    def __sigmoid_grad(self,x):
        #sigmoid gradient for backpropagation
        t = self.__sigmoid(x)
        return t*(1-t)

    def __calculate_activation(self,layer):
        #calculate the activation of specified layer
        (i,j) = self.__activations[layer - 1].shape
        self.__activations[layer] = self.__sigmoid(self.__weights[layer-1].dot(np.r_[np.ones((1,j)),self.__activations[layer-1]]))
        return

    def __calculate_dalta(self,layer):
        #calculate dalta of specified layer
        tmp = self.__weights[layer].transpose().dot(self.__dalta[layer+1])
        tmp = tmp[1:,:]
        #error : activation is value before taking sigmoid, self.__activation is sigmoid value
        self.__dalta[layer] = np.multiply(tmp,self.__sigmoid_grad(self.__activations[layer]))
        return

    def _cost(self,theta,data,y,lam):
        #reshaping input theta and setting it
        idx_start = 0
        idx_end = 0
        for n in range(self.__no_layers - 1 ):
            i,j = self.__weights[n].shape
            idx_end = idx_start + (i * j)
            self.__weights[n] = theta[idx_start:idx_end].reshape(i,j)
            idx_start = idx_end

        m,n = data.shape
        h = self.forwardprop_nn(data)
        y1 = np.subtract(1,y)
        h1 = np.subtract(1,h)
        cost1 = np.multiply(y,np.log(h))
        cost2 = np.multiply(y1,np.log(h1))
        cost = np.add(cost1,cost2)

        regualrization = 0
        for w in self.__weights:
            regualrization += np.square(w[:,1:].copy()).sum()
        
        J = ( (-1/m) * cost.sum() ) + ( lam/(2*m) ) * regualrization

        return J

    def __grad(self,theta,data,y,lam):
        #reshaping input theta and setting it
        idx_start = 0
        idx_end = 0
        for n in range(self.__no_layers - 1 ):
            i,j = self.__weights[n].shape
            idx_end = idx_start + (i * j)
            self.__weights[n] = theta[idx_start:idx_end].reshape(i,j)
            idx_start = idx_end

        m,n = data.shape
        self.backprop_nn(data,y)
        D = []
        for layer in range(self.__no_layers-1):
            tmp = (1/m) * self.__dalta[layer+1].dot(np.c_[np.ones(m),self.__activations[layer].transpose()])
            #regularization
            i,j = self.__weights[layer].shape
            reg = (lam/m) * np.c_[np.zeros(i),self.__weights[layer][:,1:]]
            D.append(  np.add(tmp,reg)  )

        #unrolling vectors
        grad = np.array([])
        for n in D:
            grad = np.concatenate([grad,n.flatten()],axis = 0)

        return grad

    def backprop_nn(self,data,y):
        h = self.forwardprop_nn(data)
        self.__dalta[-1] = np.subtract(h,y)
        for layer in reversed(range(self.__no_layers-1)):
            self.__calculate_dalta(layer)

        return
    
    def forwardprop_nn(self,data):
        #perform forward propagation 
        self.__activations[0] = data.transpose()
        for layer in range(self.__no_layers-1):
            self.__calculate_activation(layer+1)
        return self.__activations[self.__no_layers - 1]

    def predict(self,data):
        p = self.forwardprop_nn(data)
        predict_index = p.argmax(axis=0)
        predict = np.zeros(p.shape)
        for i,index in enumerate(predict_index):
            predict[index][i] = 1
        return {'predict': predict,'predict_index': predict_index}



    def Train(self):
        pass

    def architecture(self):
        pass

    def get_weights(self):
        return self.__weights
    
    def set_weights(self,weight):
        self.__weights = weight
        return

    def get_error(self):
        pass
    
    def get_activations(self):
        return self.__activations

    def test_getlayer(self):
        return self.__no_layers

    def test_cost(self,theta,data,y,lam):
        return self._cost(theta,data,y,lam)

    def test_grad(self,x):
        return self.__sigmoid_grad(x)

    def test_dalta(self):
        return self.__dalta

    def test__grad(self,theta,data,y,lam):
        return self.__grad(theta,data,y,lam)


class config():
    pass


class training_data():
    pass




