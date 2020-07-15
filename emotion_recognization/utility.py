import numpy as np
import random
import matplotlib.pyplot as plt

def display_random_image(data,labels,no = 12):
    m = data.m
    rdn = random.choices(range(m), k = no)
    images = data.data[:,rdn]
    label = data.Y[:,rdn]
    fig=plt.figure()
    for i in range(1,13):
        img = images[:,i-1].reshape(48,48)
        fig.add_subplot(3,4,i)
        plt.imshow(img)
        plt.title(labels[int(label[:,i-1][0]-1)])
    plt.show()