import numpy as np

class neural_net():
    
    def __init__(self,config):
        self.__no_layers = config.no_layers
        self.__training_data = 0
        self.__weights = []
        self.__activations= []
        self.__dalta=[]
    
    def __sigmoid(self,x):
        #sigmoid for calculting the activation value
        return 1/(1+np.exp(-x))

    def __calculate_activation(self,layer):
        #calculate the activation of specified layer
        (j,i) = np.shape(self.__activations[layer - 1])
        self.__activation[layer] = self.__sigmoid(self.__weights[layer].dot(np.r_[np.ones(i),self.__activations[layer-1]]))
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
        self.__activations[0] = data
        for layer in range(self.no_layers-1):
            self.__calculate_activation(layer+1)
        return self.__activations[self.no_layers - 1]

    def predict(self,data):
        p = self.forwardprop_nn(data)
        predict_index = p.argmax(axis=0)
        predict = np.zeros(p.shape)
        for i,index in enumerate(predict_index):
            predict[index][i] = 1
        return predict



    def Train(self):
        pass

    def get_weights(self):
        pass
    
    def get_error(self):
        pass
    
    def get_activations(self):
        pass



class config():
    pass


class training_data():
    pass




