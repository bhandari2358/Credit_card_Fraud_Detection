# Credit Card Fraud Detection Project

## Overview

This project aims to detect fraudulent credit card transactions using machine learning techniques. The dataset used includes simulated transactions spanning from 1st Jan 2019 to 31st Dec 2020, involving 1000 customers and 800 merchants. The goal is to build and evaluate models that can accurately classify transactions as legitimate or fraudulent based on transaction details.

## Dataset

The dataset consists of two main files:

- `fraudTrain.csv`: Training dataset containing transaction details.
- `fraudTest.csv`: Test dataset for evaluating model performance.

Each transaction record includes attributes such as transaction time, amount, merchant information, and whether the transaction was fraudulent (`is_fraud`).

## Project Structure

### Files:

- `credit_card_fraud_detection.py`: Main Python script for data preprocessing, model training, and evaluation.
- `fraudTrain.csv`, `fraudTest.csv`: Dataset files containing transaction details.

### Requirements:

Ensure the following Python libraries are installed:

```
pandas
numpy
scikit-learn
```

You can install them using pip:

```
pip install pandas numpy scikit-learn
```

### Running the Project:

1. **Setup:**
   - Place `credit_card_fraud_detection.py`, `fraudTrain.csv`, and `fraudTest.csv` in the same directory.

2. **Execution:**
   - Run the Python script:
     ```
     python credit_card_fraud_detection.py
     ```

   The script will preprocess the data, train the models, and evaluate their performance.

## Model Evaluation

The script evaluates the following machine learning models:

- **Logistic Regression**
- **Decision Tree Classifier**
- **Random Forest Classifier**

For each model, the evaluation metrics include:

- **Accuracy**: Proportion of correctly classified transactions.
- **F1 Score**: Harmonic mean of precision and recall.
- **Classification Report**: Detailed metrics including precision, recall, and F1-score for both classes (legitimate and fraudulent transactions).

## Results

Sample output from the script:

### Logistic Regression:

```
Accuracy: 1.00
F1 Score: 0.00
Classification Report:
              precision    recall  f1-score   support

           0       1.00      1.00      1.00    553574
           1       0.00      0.00      0.00      2145

    accuracy                           1.00    555719
   macro avg       0.50      0.50      0.50    555719
weighted avg       0.99      1.00      0.99    555719
```

### Decision Tree Classifier:

```
Accuracy: 0.99
F1 Score: 0.42
Classification Report:
              precision    recall  f1-score   support

           0       1.00      1.00      1.00    553574
           1       0.37      0.50      0.42      2145

    accuracy                           0.99    555719
   macro avg       0.68      0.75      0.71    555719
weighted avg       1.00      0.99      1.00    555719
```

### Random Forest Classifier:

```
Accuracy: 0.99
F1 Score: 0.01
Classification Report:
              precision    recall  f1-score   support

           0       1.00      1.00      1.00    553574
           1       0.02      0.01      0.01      2145

    accuracy                           0.99    555719
   macro avg       0.51      0.50      0.50    555719
weighted avg       0.99      0.99      0.99    555719
```

## Conclusion

Based on the evaluation metrics, the project assesses the performance of different models for detecting fraudulent credit card transactions. The results can guide the selection of the most effective model for deployment in real-world applications.

