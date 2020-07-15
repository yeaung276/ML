import numpy as np
from importlib.machinery import SourceFileLoader
act = SourceFileLoader("activation", "Utility/activation.py").load_module()

class basicLayer():
    def __init__(self,/,inputs,neuron,activation):
        self.fan_in = inputs
        self.fan_out = neuron
        self.W = np.zeros((neuron, inputs))
        self.b = np.zeros((neuron, 1))
        self.activation = activation
        self.activations = {
            'random' : self.__randomInitialization,
            'xavier' : self.__xavierInitialization
        }
    #initialization
    def __randomInitialization(self,epsilon):
        """Random Initialization of Weights"""
        self.W = (np.random.rand(self.W.shape) * 2 * epsilon) - epsilon
        self.b = (np.random.rand(self.b.shape) * 2 * epsilon) - epsilon

    def __xavierInitialization(self,epsilon):
        """Xavier Initialization"""
        tot = self.fan_in
        factor = np.sqrt(2/tot)
        self.W = np.random.randn(self.W.shape) * factor
        self.b = np.random.randn(self.b.shape) * factor

    def initialize(self,method='xavier',epsilon=0.5,seed=0):
        """public method for initializing the weights
           input --   method : initialization method
                                random,xavier
        """
        np.random.seed(seed)
        self.activations.get(method)(epsilon)

    #forward propagation
    def forward_propagation(self,X):
        """
        Compute forward computation
        input --   X : input activations with dimision n,m
                        n = no: of features
                        m = no: of examples
                   activation : activation function to apply
        return --  activation output A
        """
        self.Z = np.dot(self.W,X) + self.b
        self.A = act.Activations(self.Z, function = self.activation)
        self.grad_A = act.Gradients(self.Z, function = self.activation)
        return self.A

    #backward propagation
    def backward_propagation(self,dA,lambd=0):
        """
        Compute backward computation
        input --  dA : derivative of cost from the front layer with shape of n,m
                        n : number of node,
                        m : number of examples
                   lambd : lambd for regularization, 0 for no regualrization
        """
        #TODO : add regularization
        m = dA.shape[1]
        self.dZ = np.multiply(dA,self.grad_A)
        self.dW = (1/m) * np.dot(self.dZ,self.A.T)
        self.db = (1/m) * np.sum(self.dZ,axis=1,keepdims=True)
        self.dA = np.dot(self.W.T,self.dZ)
        return self.dA

    def grad(self,dA,lambd=0):
        self.backward_propagation(dA,lambd=lambd)
        return self.dw,self.db,self.dA
    


#Test Code
model = basicLayer(inputs=5,neuron=1,activation='relu')




