import numpy as np
from importlib.machinery import SourceFileLoader

LR = SourceFileLoader("linear_regression", "Supervised/linear_regression.py").load_module()

obj = LR.LinearRegression(3)

assert obj.W.shape == (1,3) and obj.b.shape == (1,1), 'err in initializion'
