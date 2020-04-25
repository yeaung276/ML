#test code for logistic regression
import logistic_regression as lc
import numpy as np

Lc = lc.Logistic_regression(3)

#testing initialization method
assert (Lc.get_weight() == np.zeros([1,3])).all() , "error in initialization method"

#testing sigmoid method
assert (Lc.test_sig(np.array([-5,-1,0,1,5])).round(4) == np.array([0.0067,0.2689,0.5,0.7311,0.9933]).round(4)).all() , "error in sigmoid method"

#testing cost function
raw_data = np.loadtxt(open("./test_data/logisticRegression/ex2data1.txt", "rb"), delimiter=",", skiprows=1)
data = raw_data[:,[0,1]]
y = raw_data[:,[2]]
(cost,grad) = Lc.test_cost_grad(data,y,0)
print('cost: {} \ngrad:{}'.format(cost,grad.ravel()))

#testing train function
result = Lc.Train(data,y,0)
print('cost: {} theta: {}'.format(result.fun,Lc.get_weight()))

y,p = Lc.predict(np.array([[78,75]]),0.5)
print('predicted: {}, probability: {}'.format(y,p))
assert y,"should be true"
