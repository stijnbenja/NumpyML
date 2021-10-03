import numpy as np

# Normalisation & standardization functions
def min_max(serie): 
    return (serie - np.min(serie)) / (np.max(serie) - np.min(serie))

def mean_normalization(serie): 
    return (serie - np.mean(serie)) / (np.max(serie) - np.min(serie))

def standardization(serie): 
    return (serie - np.mean(serie)) / np.std(serie)

def split(X, Y, train_size=0.8):
    
    cut_place = int(train_size * X.shape[1])
    
    x_train = X[ : , :cut_place ]
    x_test  = X[ : , cut_place: ]
    
    try:
        y_train = Y[ : , :cut_place ] #Generates error if it is an single array
        y_test  = Y[ : , cut_place: ]
    except:
        y_train = Y[:cut_place]
        y_test  = Y[cut_place:]
   
    return x_train, y_train, x_test, y_test
    