import numpy as np

# Regression ---------------------------------------------------------------------------------

def MAE(y_real, y_pred):
    difference_array = y_real - y_pred
    return np.mean(np.abs(difference_array))

def MSE(y_real, y_pred):
    difference_array = y_real - y_pred
    return np.mean(np.power(difference_array, 2))

def MSLE(y_real, y_pred):
    return np.mean(np.power(np.log(y_real+1) - np.log(y_pred+1), 2))
    
# Binary classification ----------------------------------------------------------------------

def binary_crossentropy(y_real, y_pred):
    extra = 1e-10 #Against division by zero
    return -np.mean(y_real * np.log(y_pred + extra) + (1-y_real) * np.log(1-y_pred + extra))

def hinge(y_real, y_pred):
    pass

def squared_hinge(y_real, y_pred):
    pass

# Multi-class classification -----------------------------------------------------------------

def multiclass_entropy(y_real, y_pred):
    pass

def sparse_multiclass_cross_entropy(y_real, y_pred):
    pass

def kullback_leibler_divergence(y_real, y_pred):
    pass