import numpy as np
from numba import njit

# Normalisation & standardization functions
@njit(fastmath=True, cache=True)
def min_max(serie): 
    return (serie - np.min(serie)) / (np.max(serie) - np.min(serie))

@njit(fastmath=True, cache=True)
def mean_normalization(serie): 
    return (serie - np.mean(serie)) / (np.max(serie) - np.min(serie))

@njit(fastmath=True, cache=True)
def standardization(serie): 
    return (serie - np.mean(serie)) / np.std(serie)