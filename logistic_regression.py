import numpy as np
import scipy.optimize as op

class Logistic_regression():
    def __init__(self,no_feature):
        self.__j=0
        self.__theta = np.zeros([1,no_feature])

    def __sigmoid(self,x):
        #sigmoid for calculting the activation value
        return 1/(1+np.exp(-x))

    def __cost(self,theta,T_data,y,lam):
        (i,j) = self.__theta.shape
        theta = theta.reshape(1,j)
        #adding bias term ones to the data
        (i,j) = T_data.shape
        m = i
        p = np.c_[np.ones(i),T_data]
        #dot product the traning data with theta and take sigmoid to get the hypothesis
        p = self.__sigmoid(p.dot(theta.transpose()))

        p1 = np.subtract(1,p)
        y1 = np.subtract(1,y)
        cost1 = np.multiply(y,np.log(p))
        cost2 = np.multiply(y1,np.log(p1))
        cost = np.add(cost1,cost2)

        regualrization = theta.copy()
        regualrization[:,0] = 0
        regualrization = np.square(regualrization)

        J = (-1/m) * cost.sum() + (lam/(2*m)) * regualrization.sum()
        
        self.__J = J

        return J

    def __grad(self,theta,T_data,y,lam):
        (i,j) = self.__theta.shape
        theta = theta.reshape(1,j)

        m,n = T_data.shape
        data = np.c_[np.ones(m),T_data]
        p = self.__sigmoid(data.dot(theta.transpose()))

        err = np.subtract(p,y).transpose()
        reg_theta = theta.transpose().copy()
        #removing bias term
        reg_theta[:,0] = 0

        grad = (1/m) * err.dot(data).transpose() + (lam/m) * reg_theta

        return grad.flatten()


    def Train(self,data,y,lam):
        print('Training Model......')
        self.__iter = 0
        Result = op.minimize(
                fun = self.__cost,
                x0 = self.__theta.copy(),
                args = (data,y,lam),
                method = 'BFGS',
                jac = self.__grad,
                options = { 'maxiter': 500,'disp': True },
                callback=self.__callback
                )
        self.__theta = Result.x.reshape(self.__theta.shape)
        return Result

    def __callback(self,xi):
        i=self.__J
        j=self.__iter
        print('iter: [%d] cost: [%f]\r'%(j,i), end="")
        self.__iter+=1


    def get_weight(self):
        return self.__theta

    def set_weight(self,weight):
        self.__theta = weight

    def predict(self,data,threshold):
        data = data.copy()
        i,j = data.shape
        data = np.c_[np.ones(i),data]
        p = self.__sigmoid(self.__theta.dot(data.transpose()))
        return p > threshold,p

    def getCost(self,data,y):
        return self.__cost(self.__theta,data,y,0)
    
    def test_sig(self,x):
        return self.__sigmoid(x)

    def test_cost_grad(self,data,y,lam):
        return self.__cost(self.__theta,data,y,lam),self.__grad(self.__theta,data,y,lam)

    



