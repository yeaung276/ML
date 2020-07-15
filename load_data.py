import os as os
import numpy as np
from scipy.io import savemat
from PIL import Image

loaded_data = {}
size = 2304
base_path = 'faceData/CK+48/'

def get_image_path(label,file_name):
    return base_path+label+'/'+file_name
labels = os.listdir(base_path)
#save the data in .mat format
for label in labels:
    file_list = os.listdir(base_path + label)
    print('{} : {}'.format(label,len(file_list)))
    X = np.zeros((size,1))
    for file_name in file_list:
        image = np.array(Image.open(get_image_path(label,file_name)).getdata()).reshape(size,1)
        X = np.hstack((X,image))
    loaded_data[label] = X[:,1:]

savemat('faceData/face_data.mat',loaded_data)

#save the intire training set and data to .mat format
X = np.zeros((size,1))
Y = np.zeros((1,1))
for i,label in enumerate(labels):
    t = loaded_data[label]
    X = np.hstack((X,t))
    Y = np.hstack((Y,(i+1)*np.ones((1,t.shape[1]))))
savemat('data_set.mat',{
    'X' : X[:,1:],
    'Y' : Y[:,1:],
    'labels' : labels
})
