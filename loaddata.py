#! /bin/env python3

import pickle
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize

def load_pkl(fname):

    with open(fname,'rb') as f:
        return pickle.load(f)


def save_pkl(fname, obj):

    with open(fname,'wb') as f:
        pickle.dump(obj,f)


if __name__ == "__main__":
    data = np.array(load_pkl("train_data.pkl"))
    labels = np.load("finalLabelsTrain.npy")
    concatenated_data = np.vstack((data,labels)).T
    for i in range(6400):
        plt.figure()
        data1 = resize(concatenated_data[i][0], (49, 50))
        plt.imshow(data1)
        plt.show()
        print(concatenated_data[i][0])