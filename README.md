Credit Card Fraud Detection Project
Overview

This project aims to detect fraudulent credit card transactions using machine learning algorithms. The dataset contains simulated transaction data spanning from 1st Jan 2019 to 31st Dec 2020. It includes transactions from 1000 customers across 800 merchants.
Objective

Build and evaluate machine learning models to classify transactions as fraudulent or legitimate.
Dataset

The dataset (fraudTrain.csv and fraudTest.csv) includes the following fields:

    amt: Transaction amount
    merchant: Merchant name
    category: Transaction category
    city: Merchant city
    state: Merchant state
    job: Customer's occupation
    is_fraud: Binary indicator (1 for fraudulent, 0 for legitimate)

Algorithms Used

    Logistic Regression
    Decision Trees
    Random Forests

Code Structure
Prerequisites

    Python 3
    Libraries: numpy, pandas, scikit-learn

Installation

    Clone the repository:

    bash

git clone https://github.com/your_username/credit-card-fraud-detection.git

Install dependencies:

bash

    pip install -r requirements.txt

Usage

    Data Preparation:
        Ensure fraudTrain.csv and fraudTest.csv are in the project directory.

    Model Training and Evaluation:
        Run credit_card_fraud_detection.py to train and evaluate models.
        Adjust parameters or algorithms in the script as needed.

Evaluation Metrics

    Accuracy: Percentage of correctly predicted transactions.
    F1 Score: Harmonic mean of precision and recall, useful for imbalanced classes.

Results

Results of model evaluation on the test set:

    Logistic Regression:
        Accuracy: 1.00
        F1 Score: 0.00

    Decision Tree Classifier:
        Accuracy: 0.99
        F1 Score: 0.42

    Random Forest Classifier:
        Accuracy: 0.99
        F1 Score: 0.01

Conclusion

The project demonstrates various machine learning techniques for detecting fraudulent credit card transactions. Further improvements can be made by fine-tuning model parameters or exploring additional algorithms.
