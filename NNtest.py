import scipy as sp
import scipy.io
import numpy as np
import neural_net 

weights = sp.io.loadmat('./test_data/nn/ex3weights.mat')
weight1 = np.array(weights['Theta1'])
weight2 = np.array(weights['Theta2'])
weight = [weight1,weight2]

#testing setter and getter of weight and no: of layer initialization
#testing rand initializing
NN = neural_net.neural_net({'architecture': [10,1],'rand_range': 0.7})
for w in NN.get_weights():
    assert (w>-0.7).all() and (w<0.7).all() , 'rand range implementation error'
NN = neural_net.neural_net({'architecture': [400,25,1]})
assert len(NN.get_activations()) == 3 , 'error in initialization activation'
assert (NN.get_weights()[0].shape == (25,401)) & (NN.get_weights()[1].shape == (1,26)) , 'error in initializing weight'
NN.set_weights(weight)
assert (NN.get_weights()==weight) & (NN.test_getlayer() == 3) , 'error in getting or setting weight'

#testing forward propagation method
data = sp.io.loadmat('./test_data/nn/ex3data1.mat')
X = np.array(data['X'])
y = np.array(data['y'])
assert round(NN.forwardprop_nn(X).sum(),1) == 5023.2 , 'error in forward propagation method'
assert (((NN.predict(X)['predict_index'] + 1).reshape(5000,1) == y).mean()) >= 0.9752 , 'error in prediction function'

#testing back propagation
weights = sp.io.loadmat('./test_data/nn/ex4weights.mat')
weight1 = np.array(weights['Theta1'])
weight2 = np.array(weights['Theta2'])
weight = [weight1,weight2]
NN = neural_net.neural_net({'architecture': [400,25,10]})
NN.set_weights(weight)

data = sp.io.loadmat('./test_data/nn/ex4data1.mat')
X = np.array(data['X'])
y = np.array(data['y'])
theta = np.array([])
y_mod = np.zeros((10,5000))

#testing cost
for i,n in enumerate(y):
    y_mod[n-1,i] = 1
for n in weight:
    theta = np.concatenate([theta,n.flatten()],axis = 0) 
assert round(NN.test_cost(theta,X,y_mod,0),3)==0.288, 'should be 0.288'
assert round(NN.test_cost(theta,X,y_mod,1),3)==0.384, 'should be 0.384'

#testing sigmoid gradient
tD = np.array([-1,-0.5,0,0.5,1])
assert (NN.test_grad(tD).round(6)==np.array([0.196612,0.235004,0.250000,0.235004,0.196612])).all() ,'error in sigmoid gradient'