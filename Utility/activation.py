import numpy as np


def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_grad(x):
    t = sigmoid(x)
    return t * (1 - t)

def identity(x):
    return x

def identity_grad(x):
    return np.ones(x.shape)

def tanh(x):
    return np.tanh(x)

def tanh_grad(x):
    t = tanh(x)
    return 1 - np.power(t,2)

def relu(x):
    return np.maximum(0,x)

def relu_grad(x):
    t = np.ones(x.shape)
    t[x <= 0] = 0
    return t

def leaky_relu(x, c = 0.01):
    return np.maximum(c*x,x)  

def leaky_relu_grad(x, c = 0.01):
    t = np.ones(x.shape)
    t[x <= 0] = c
    return t

Grads = {
    'sigmoid': sigmoid_grad,
    'identity': identity_grad,
    'tanh': tanh_grad,
    'relu': relu_grad,
    'leaky_relu': leaky_relu_grad
}
Act_functions = {
    'sigmoid': sigmoid,
    'identity': identity,
    'tanh': tanh,
    'relu': relu,
    'leaky_relu': leaky_relu
}
def Activations(x, function = 'relu'):
    return Act_functions.get(function)(x)

def Gradients(x, function = 'relu'):
    return Grads.get(function)(x)

