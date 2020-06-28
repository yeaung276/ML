
import scipy.stats 
from importlib.machinery import SourceFileLoader

data = SourceFileLoader("data", "Utility/data.py").load_module()

class Anomoly(data.Data):

    def __init__(self,X):
        n,m = X.shape
        super().__init__(X,m)
    
    def anomolyDetect(self,X, destibution = 'gussian', threadshold = 0.004):
        val = None
        warning = ''
        if(destibution == 'gussian'):
            val = self.oneDimensionalGussian(X)
        elif(destibution == 'multigussian'):
            val = self.multiDimensionalgussian(X)
        else:
            print('not found')
        print('need something in threadsholding or somewhat')
        if(val < threadshold):
            warning = 'anomoly'
        return val, warning

    def oneDimensionalGussian(self,X):
        n,m = X.shape
        val = 1
        for i in range(n):
            val *= scipy.stats.norm(self.mu[i,0], self.sigma[i,0]).pdf(X[i,0])
        return val

    def multiDimensionalgussian(self,X):
        print('to be implemented')
        

