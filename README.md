# Credit Card Fraud Detection
Overview

This project implements machine learning models to detect fraudulent credit card transactions. It includes data preprocessing, model training, and evaluation using Python and scikit-learn.
Files Included

    fraudTrain.csv: Training dataset containing transaction information.
    fraudTest.csv: Test dataset for evaluating model performance.
    credit_card_fraud_detection.py: Python script for data preprocessing, model training, and evaluation.

Prerequisites

Ensure you have Python 3.x installed along with the following libraries:

    NumPy
    pandas
    scikit-learn

You can install the required libraries using pip:

bash

pip install numpy pandas scikit-learn

Running the Script

    Download the Datasets:
        Place fraudTrain.csv and fraudTest.csv in the same directory as credit_card_fraud_detection.py.

    Execute the Script:
    Run the Python script from the command line:

    bash

    python credit_card_fraud_detection.py

Data Preprocessing

    Cleaning: Dropping unnecessary columns (Unnamed: 0, trans_date_trans_time, cc_num, first, last, street, dob, trans_num).
    Feature Engineering: Limiting unique categorical values for high-cardinality features (merchant, category, city, state, job).
    One-Hot Encoding: Converting categorical variables into binary vectors for model compatibility.

Models Implemented

    Logistic Regression:
        Used for baseline classification with linear decision boundary.
    Decision Tree Classifier:
        Decision-making tree structure for non-linear decision boundaries.
    Random Forest Classifier:
        Ensemble of decision trees to improve robustness and accuracy.

Evaluation Metrics

    Accuracy: Percentage of correct predictions out of total predictions.
    F1 Score: Harmonic mean of precision and recall, useful for imbalanced classes.

Results
Logistic Regression

    Accuracy: 1.00
    F1 Score: 0.00
    Classification Report: Precision and recall statistics for fraud and non-fraud classes.

Decision Tree Classifier

    Accuracy: 0.99
    F1 Score: 0.42
    Classification Report: Detailed breakdown of model performance metrics.

Random Forest Classifier

    Accuracy: 0.99
    F1 Score: 0.01
    Classification Report: Evaluation metrics highlighting strengths and weaknesses.

Conclusion

The models demonstrate varying levels of effectiveness in detecting fraudulent transactions. Further enhancements such as hyperparameter tuning or feature engineering could potentially improve performance.
