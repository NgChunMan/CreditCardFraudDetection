# Credit Card Fraud Detection

## Overview

This project implements a machine learning solution to detect fraudulent transactions in credit card datasets. Using logistic regression with batch and stochastic gradient descent (SGD), the model predicts whether a transaction is fraudulent or not. The dataset used consists of legitimate and fraudulent transactions.

## Features
- Logistic Regression Implementation:
Logistic regression is implemented from scratch, providing a deeper understanding of its underlying mechanics and flexibility for modifications.
- Gradient Descent Optimization:
Both batch gradient descent and stochastic gradient descent (SGD) are employed to train the model efficiently, even with large datasets.
- Cost Function and Weight Updates:
Includes a custom implementation of cross-entropy loss for evaluating model performance and weight update logic to optimize predictions.
- Fraud Detection:
Classifies transactions as either legitimate (0) or fraudulent (1).

## Setup Instructions

1. Clone the Repository:
```
git clone https://github.com/NgChunMan/CreditCardFraudDetection.git
cd CreditCardFraudDetection
```

2. Install required Dependencies:
```
pip install -r requirements.txt
```

3. Download the following dataset and place them in the `data/` directory:
- [credit_card.csv](https://drive.google.com/file/d/1DXAtZnr-mrHccmMX6k1NRssRz2T889G3/view?usp=drivesdk)

4. Run the main script to train the model and classify transactions:
```
python main.py
```

## Testing
Unit tests are provided to validate the implementation. Run the tests using pytest:
```
pytest tests/
```

## Results
Example Output:
`Predictions: [0 1 0 ... 1 0 1]`

It outputs an array of predictions for all transactions in the dataset.
