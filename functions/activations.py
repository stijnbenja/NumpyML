import numpy as np

'''

'''

#SIGMOID
def sigmoid(x):
    ceiling = 100 #Against multiplication overflow
    big_out = np.where(x > ceiling, ceiling, x)
    small_out = np.where(big_out < -ceiling, -ceiling, big_out)
    return 1 / (1 + np.exp(-small_out))

def dSigmoid(x):
    return x * (1-sigmoid(x))

#TANH
def tanh(x):
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

def dTanh(x):
    return 1 - np.power(tanh(x), 2)

#RELU
def relu(x):
    np.max(0,x)
    
def dRelu(x):
    return 1 if (x > 0) else 0

#ELU
def elu(x, alpha):
    return x if (x >= 0) else alpha * (np.exp(x) -1)
    
def dElu(x, alpha):
    return 1 if (x >= 0) else alpha*np.exp(x)

#SOFTMAX
def softmax(x):
    k = np.exp(x - np.max(x))
    return k / np.sum(k)#, axis=1)

def dSoftmax(x):
    