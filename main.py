import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from src.logistic_regression_stochastic_gradient_descent import logistic_regression_stochastic_gradient_descent
from src.model import logistic_regression_classification

# Set print options to display all classifications
np.set_printoptions(threshold=np.inf)

# Load dataset
data_path = "data/credit_card.csv"
credit_df = pd.read_csv(data_path)

# Preprocess data
X = credit_df.iloc[:, :-1].to_numpy()
y = credit_df.iloc[:, -1].to_numpy()

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train model
weights = logistic_regression_stochastic_gradient_descent(X_scaled, y)

# Predict results
predictions = logistic_regression_classification(X_scaled, weights)

print("Predictions:", predictions)
