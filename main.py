import pandas as pd
from src.logistic_regression_stochastic_gradient_descent import logistic_regression_stochastic_gradient_descent
from src.model import logistic_regression_classification

# Load dataset
data_path = "data/credit_card.csv"
credit_df = pd.read_csv(data_path)

# Preprocess data
X = credit_df.iloc[:, :-1].to_numpy()
y = credit_df.iloc[:, -1].to_numpy()

# Train model
weights = logistic_regression_stochastic_gradient_descent(X, y)

# Predict results
predictions = logistic_regression_classification(X, weights)

print("Predictions:", predictions)
