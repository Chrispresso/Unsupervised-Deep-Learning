import numpy as np
import pandas as pd
from sklearn.utils import shuffle


def relu(x):
    return x * (x > 0)

def error_rate(p, t):
    return np.mean(p != t)

def get_kaggle_MNIST():
    train = pd.read_csv('train.csv').as_matrix().astype(np.float32)
    train = shuffle(train)

    Xtrain = train[:-1000, 1:] / 255.0
    Ytrain = train[:-1000, 0].astype(np.int32)

    Xtest = train[-1000:, 1:] / 255.0
    Ytest = train[-1000:, 0].astype(np.int32)

    return Xtrain, Ytrain, Xtest, Ytest

def init_weights(shape):
    return np.random.randn(*shape) / np.sqrt(sum(shape))