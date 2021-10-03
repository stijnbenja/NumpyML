import numpy as np


binary_crossentropy = lambda y_real, y_pred: -np.mean(y_real * np.log(y_pred + 1e-10) + (1-y_real) * np.log(1-y_pred + 1e-10))
