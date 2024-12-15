# Credit Card Fraud Detection Project

## Overview

This project uses a machine learning solution to detect fraudulent credit card transactions. It implements logistic regression from scratch, optimized with batch and stochastic gradient descent, to classify transactions as either fraudulent or legitimate. The project also tackles imbalanced dataset challenges and ensures flexibility for future enhancements.

## Features
- Logistic Regression Implementation:
Logistic regression is implemented from scratch, providing a deeper understanding of its underlying mechanics and flexibility for modifications.
- Gradient Descent Optimization:
Both batch gradient descent and stochastic gradient descent (SGD) are employed to train the model efficiently, even with large datasets.
- Cost Function and Weight Updates:
Includes a custom implementation of cross-entropy loss for evaluating model performance and weight update logic to optimize predictions.

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

4. Run the main script:
```
python main.py
```

## Testing
Unit tests are provided to validate the implementation of the gradient descent algorithm and other utilities. Run the tests using pytest:
```
pytest tests/
```

## Results
...

