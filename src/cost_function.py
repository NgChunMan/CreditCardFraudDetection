import numpy as np

def cost_function(X: np.ndarray, y: np.ndarray, weight_vector: np.ndarray):
    '''
    Cross entropy error for logistic regression
    '''    
    # Machine epsilon for numpy `float64` type
    eps = np.finfo(np.float64).eps

    m = X.shape[0]
    Xw = np.matmul(X, weight_vector)
    e_Xw = np.exp(-1 * Xw)
    y_pred = 1 / (1 + e_Xw)

    cross_entropy = -y * np.log(y_pred + eps) - (1 - y) * np.log(1 - y_pred + eps)
    error_cost = np.sum(cross_entropy) / m
    return error_cost
