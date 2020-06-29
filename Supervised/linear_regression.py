import numpy as np
from importlib.machinery import SourceFileLoader

linearLayer = SourceFileLoader("linear_layer", "Supervised/linear_layer.py").load_module()


class LinearRegression(linearLayer.linear_layer):
    def __init__(self,layer,output = 1):
        super().__init__(layer,output)
        