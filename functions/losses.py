import numpy as np

def binary_crossentropy(y_real, y_pred):
    extra = 1e-10 #Against division by zero
    return -np.mean(y_real * np.log(y_pred + extra) + (1-y_real) * np.log(1-y_pred + extra))