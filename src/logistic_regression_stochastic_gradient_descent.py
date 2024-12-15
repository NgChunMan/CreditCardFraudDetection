import numpy as np
import pandas as pd
from src.cost_function import cost_function

def logistic_regression_stochastic_gradient_descent(X_train: np.ndarray, y_train: np.ndarray, max_num_iterations: int=250, threshold: np.float64=0.05, alpha: np.float64=1e-5, seed: int=43) -> np.ndarray:
    '''
    Initialize your weight to zeros. Write a terminating condition, and run the weight update for some iterations.
    Get the resulting weight vector.
    '''
    num_of_features = X_train.shape[1]
    weight_vector = np.zeros(num_of_features)
    m = X_train.shape[0]
    error_value = cost_function(X_train, y_train, weight_vector)
    np.random.seed(seed)

    for i in range(max_num_iterations):
        if error_value <= threshold:
            return weight_vector
        
        random_sample_index = np.random.choice(m, size=1)[0]
        random_sample = X_train[random_sample_index]
        true_value_of_random_sample = y_train[random_sample_index]

        X_train_w = np.matmul(random_sample, weight_vector)
        e_X_train_w = np.exp(-X_train_w)
        y_pred = 1 / (1 + e_X_train_w)

        difference_pred_true = y_pred - true_value_of_random_sample

        partial_derivative_w = difference_pred_true * random_sample
        partial_derivative_w = alpha * partial_derivative_w
        weight_vector = weight_vector - partial_derivative_w
        error_value = cost_function(X_train, y_train, weight_vector)

    return weight_vector
