import numpy as np

def logistic_regression_classification(X: np.ndarray, weight_vector: np.ndarray, prob_threshold: np.float64=0.5):
    '''
    Do classification task using logistic regression.
    '''
    Xw = np.matmul(X, weight_vector)
    e_Xw = np.exp(-Xw)
    y_pred = 1 / (1 + e_Xw)
    classification_result = np.where(y_pred > prob_threshold, 1, 0)

    return classification_result
  
