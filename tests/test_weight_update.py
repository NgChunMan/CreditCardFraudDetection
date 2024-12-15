import numpy as np
import pandas as pd
from src.weight_update import weight_update

def test_weight_update():
  data1 = [[111.1, 10, 0], [111.2, 20, 0], [111.3, 10, 0], [111.4, 10, 0], [111.5, 10, 0], [111.6, 10, 1],[111.4, 10, 0], [111.5, 10, 1], [111.6, 10, 1]]
  df1 = pd.DataFrame(data1, columns = ['V1', 'V2', 'Class'])
  X1 = df1.iloc[:, :-1].to_numpy()
  y1 = df1.iloc[:, -1].to_numpy()
  w1 = np.transpose([2.2000, 12.20000])
  a1 = 1e-5
  nw1 = np.array([2.199,12.2])
  
  assert np.array_equal(np.round(weight_update(X1, y1, a1, w1), 3), nw1)
