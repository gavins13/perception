import pickle as cpk
import numpy as np
import random
import os


def load_data():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    filename= dir_path + '/rmnist_expanded_10.pkl'
    print("Loading data")
    with open(filename, 'rb') as fname:
        print("Opened data successfully")
        mnist = cpk.load(fname, encoding='latin1') # latin1 due to incompatibility between pickle in python2 and python3


    mnist_data = np.array(mnist[0][0]) # (900, 784)
    mnist_labels = np.array(mnist[0][1]) # (900,)
    del mnist
    #mnist_test_data = mnist[1][0] # (10000, 784)
    #mnist_test_labels = mnist[1][1] # (10000,)

    data_idx = random.sample(range(900), 250)
    train_data_idx = data_idx[0:200]
    test_data_idx = data_idx[200:250]

    mnist_train_data = mnist_data[train_data_idx, :]
    mnist_train_labels = mnist_labels[train_data_idx]
    mnist_test_data = mnist_data[test_data_idx, :]
    mnist_test_labels = mnist_data[test_data_idx]

    mnist_train_data = np.reshape(mnist_train_data, [200, 28, 28])
    mnist_test_data = np.reshape(mnist_test_data,[50, 28, 28])

    print("Finished loading reduced-size MNIST dataset (200 training, 50 test)")

    return mnist_train_data, mnist_train_labels, mnist_test_data, mnist_test_labels
