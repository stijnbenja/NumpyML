import numpy as np
from numba import njit

@njit(fastmath=True, cache=True)
def sigmoid(x):
    ceiling = 100 #Against multiplication overflow
    big_out = np.where(x > ceiling, ceiling, x)
    small_out = np.where(big_out < -ceiling, -ceiling, big_out)
    return 1 / (1 + np.exp(-small_out))

@njit(fastmath=True, cache=True)
def dSigmoid(x):
    return x * (1-sigmoid(x))