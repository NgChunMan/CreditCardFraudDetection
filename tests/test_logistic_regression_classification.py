import numpy as np
import pandas as pd
from src.model import logistic_regression_classification

def test_logistic_regression_classification():
  data1 = [[111.1, 10, 0], [111.2, 20, 0], [111.3, 10, 0], [111.4, 10, 0], [111.5, 10, 0], [211.6, 80, 1],[111.4, 10, 0], [111.5, 80, 1], [211.6, 80, 1]]
  df1 = pd.DataFrame(data1, columns = ['V1', 'V2', 'Class'])
  X1 = df1.iloc[:, :-1].to_numpy()
  y1 = df1.iloc[:, -1].to_numpy()
  w1 = np.transpose([-0.000002, 0.000003])
  expected1 = np.transpose([0, 0, 0, 0, 0, 0, 0, 1, 0])
  result1 = logistic_regression_classification(X1, w1)
  
  assert result1.shape == expected1.shape and (result1 == expected1).all()
