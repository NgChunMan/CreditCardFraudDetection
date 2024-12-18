import numpy as np
import pandas as pd
from src.cost_function import cost_function

def test_cost_function():
  data1 = [[111.1, 10, 0], [111.2, 20, 0], [111.3, 10, 0], [111.4, 10, 0], [111.5, 10, 0], [111.6, 10, 1], [111.4, 10, 0], [111.5, 10, 1], [111.6, 10, 1]]
  df1 = pd.DataFrame(data1, columns = ['V1', 'V2', 'Class'])
  X1 = df1.iloc[:, :-1].to_numpy()
  y1 = df1.iloc[:, -1].to_numpy()
  w1 = np.transpose([0.002, 0.1220])
  
  assert np.round(cost_function(X1, y1, w1), 5) == np.round(1.29333, 5)
