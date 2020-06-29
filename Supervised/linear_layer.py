import numpy as np
from importlib.machinery import SourceFileLoader

activation = SourceFileLoader('activation','/Utility/activation.py').load_module()

class linear_layer():
    def __init__(self, nodes, output = 1):
        self.W = np.zeros((output,nodes))
        self.b = np.zeros((output, 1))

    def __linear(self, X):
        """ X : input data with the shape n,m 
                 where m = no: of example
                       n = no: of feature
            return : none
            effect : set linear output as Z"""

        W = self.W
        b = self.b
        self.Z = np.dot(W,X) + b

    def __activation(self, function = 'identity'):
        """compute the non-linear activation output from Z but use 
        identity activation function for linear regression
        effect : set activation output as A"""

        self.A = activation.Activations(self.Z, function = function)
        self.A_prime = activation.Activations(self.Z, function = function)

    
    def forward_pass(self,X):
        """compute the forward computation
        X : input data with shape of n,m
            where m = no: of example
                  n = no: of feature
        return : activation ouput of forward computation"""

        self.__linear(X)
        self.__activation()
        return self.A

    def __compute_L1_loss(self,X,Y):
        #compute level 1 loss
        self.__L1_loss = forward_pass(X) - Y
    
    #TODO: add regularization
    def cost(self,X,Y,lambd = 0):
        """ compute the cost of the model
        X : input data with shape n,m
            where m = no: of example
                  n = no: of feature
        Y : input label with shape n',m
            where n'= no: of output node
                  m = no: of example
        lambd : regularization parameter
        return : cost """

        n,m = X.shape
        self.__compute_L1_loss(X,Y)
        L1_loss = self.__L1_loss
        J = 1/(2*m) * np.dot(L1_loss,L1_loss.T)
        return J

    #TODO: add regularization
    def grads(self,X,Y,lambd = 0):
        """compute the gradient of this layer
        X : input data with shape n,m
            where m = no: of example
                  n = no: of feature
        Y : input label with shape n',m
            where n'= no: of output node
                  m = no: of example 
        lambd : regularization parameter
        return : a tuple of flattened gradient vector, back dZ gradient 
                 and dictionary containing both"""

        n,m = X.shape
        L1_loss = self.__L1_loss
        W = self.W
        self.dW = (1/m) * np.dot(L1_loss, X.T).T
        self.db = (1/m) * np.sum(L1_loss)
        self.dZ = (1/m) * np.dot(L1_loss, W.T).T * self.A_prime
        grads = np.concatenate(self.dW.flatten(),self.db.flatten())
        backprop_out = {
            'grads' : grads,
            'dZ'    : self.dZ
        }

        return grads,self.dZ,backprop_out

    


    
        