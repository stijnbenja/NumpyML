import numpy as np

# Normalisation & standardization functions
def min_max(serie): return (serie - np.min(serie)) / (np.max(serie) - np.min(serie))
def mean_normalization(serie): return (serie - np.mean(serie)) / (np.max(serie) - np.min(serie))
def standardization(serie): return (serie - np.mean(serie)) / np.std(serie)
