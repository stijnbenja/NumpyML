import numpy as np


def sigmoid(x):
    big_out = np.where(x > 100, 100, x)
    small_out = np.where(big_out < -100, -100, big_out)
    return 1 / (1 + np.exp(-small_out))

def dSigmoid(x):
    return x * (1-sigmoid(x))
