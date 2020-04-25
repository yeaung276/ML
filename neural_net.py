import numpy as np

class neural_net():
    
    def __init__(self,config):
        self.__no_layers = len(config['architecture'])
        self.__training_data = 0
        self.__weights = []
        self.__activations= [ np.zeros((n,1)) for n in config['architecture'] ]
        self.__dalta=[ np.zeros((n,1)) for n in config['architecture']]
    
    def __sigmoid(self,x):
        #sigmoid for calculting the activation value
        return 1/(1+np.exp(-x))

    def __calculate_activation(self,layer):
        #calculate the activation of specified layer
        (i,j) = self.__activations[layer - 1].shape
        self.__activations[layer] = self.__sigmoid(self.__weights[layer-1].dot(np.r_[np.ones((1,j)),self.__activations[layer-1]]))
        return

    def _cost(self,data,y,m,lam):
        h = self.forwardprop_nn(data)
        y1 = np.subtract(1,y)
        h1 = np.subtract(1,h)
        cost1 = np.multiply(y,h)
        cost2 = np.multiply(y1,h1)
        cost = np.add(cost1,cost2)
        regualrization = np.square(self.__weights[:,:,[1,-1]])
        J = ( (-1/m) * cost.sum() ) + ( lam/(2*m) ) * regualrization.sum()
        return J

    def backprop_nn(self):
        pass
    
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


class config():
    pass


class training_data():
    pass




