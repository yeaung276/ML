import numpy as np
from importlib.machinery import SourceFileLoader

activation = SourceFileLoader("activation", "Utility/activation.py").load_module()

a = np.array([[-1,-2,-3],[4,5,0]])
print(activation.Activations(a, function = 'leaky_relu'))
print(activation.Gradients(a, function = 'leaky_relu'))
