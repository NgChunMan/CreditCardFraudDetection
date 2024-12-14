import numpy as np

def weight_update(X: np.ndarray, y: np.ndarray, alpha: np.float64, weight_vector: np.ndarray) -> np.ndarray:
    '''
    Do the weight update for one step in gradient descent
    '''

    m = X.shape[0]
    random_sample_index = np.random.choice(m, size=1)[0]
    random_sample = X[random_sample_index]
    true_value_of_random_sample = y[random_sample_index]

    random_weight_index = np.random.choice(len(weight_vector), size=1)[0]
    random_weight = weight_vector[random_weight_index]

    Xw = np.matmul(random_sample, weight_vector)
    e_Xw = np.exp(-Xw)
    y_pred = 1 / (1 + e_Xw)

    difference_pred_true = y_pred - true_value_of_random_sample
    partial_derivative_w = difference_pred_true * random_sample[random_weight_index]
    partial_derivative_w = alpha * partial_derivative_w

    weight_vector[random_weight_index] = random_weight - partial_derivative_w
    return weight_vector
