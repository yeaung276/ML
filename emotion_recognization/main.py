import numpy as np
import scipy as sp
import scipy.io
import utility 
from importlib.machinery import SourceFileLoader

data = SourceFileLoader("data", "Utility/data.py").load_module()

faces = sp.io.loadmat('data_set.mat')

datas = data.Data(faces['X'],981,label = faces['Y'])
print(datas.info())
datas.featureScale(Scaletype = 'max_norm')
datas.shuffle()
utility.display_random_image(datas,faces['labels'])



