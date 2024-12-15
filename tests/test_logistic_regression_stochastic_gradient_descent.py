import numpy as np
import pandas as pd
from src.cost_function import cost_function
from src.logistic_regression_stochastic_gradient_descent import logistic_regression_stochastic_gradient_descent

def test_logistic_regression_stochastic_gradient_descent():
    data1 = [[111.1, 10, 0], [111.2, 20, 0], [111.3, 10, 0], [111.4, 10, 0], [111.5, 10, 0], [211.6, 80, 1],[111.4, 10, 0], [111.5, 80, 1], [211.6, 80, 1]]
    df1 = pd.DataFrame(data1, columns = ['V1', 'V2', 'Class'])
    X1 = df1.iloc[:, :-1].to_numpy()
    y1 = df1.iloc[:, -1].to_numpy()
    expected1 = cost_function(X1, y1, np.transpose(np.zeros(X1.shape[1])))

    assert cost_function(X1, y1, logistic_regression_stochastic_gradient_descent(X1, y1)) < expected1
