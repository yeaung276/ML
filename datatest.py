import data as data
import anomoly_detect as ano
import numpy as np

X = np.array([[1,2,3,4,5,6,7,8,9,10,11],[1,3,5,7,9,11,9,7,5,3,1]])
Y = np.array([[1,2,3,4,5,6,7,8,9,10,11]])
obj = data.Data(X,11)
#hell = obj.Normalize()
#obj.featureScale()
#print(obj.mu)
d = ano.Anomoly(X)
#print(d.anomolyDetect(np.array([[1],[1]])))
#batches = obj.mini_batch(batchsize=3)
#obj2 = data.Data(X,11,label=Y)
#batches = obj2.mini_batch(batchsize=3)
#print(batches[0].Y)
#obj2.shuffle()
#print(obj2.data)
#print(obj2.Y)
#obj2.info()
