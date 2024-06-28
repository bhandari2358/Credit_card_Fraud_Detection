import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Function to reduce the number of unique categories
def limit_unique_values(df, column, max_unique_values=100):
    value_counts = df[column].value_counts()
    top_values = value_counts.index[:max_unique_values]
    df[column] = df[column].where(df[column].isin(top_values), 'Other')
    return df

# Load the training dataset
df_train = pd.read_csv('fraudTrain.csv')

# Drop unnecessary columns
df_train.drop(columns=["Unnamed: 0", "trans_date_trans_time", "cc_num", "first", "last", "street", "dob", "trans_num"], inplace=True)

# Select a subset of data for training (you can adjust this as needed)
data = df_train.head(n=20000)

# Limit unique values for high-cardinality categorical features
high_cardinality_columns = ['merchant', 'category', 'city', 'state', 'job']
for column in high_cardinality_columns:
    data = limit_unique_values(data, column)

# One-hot encode categorical variables
data_processed = pd.get_dummies(data)

# Split features and target variable
X_train = data_processed.drop(columns='is_fraud', axis=1)
y_train = data_processed['is_fraud']

# Load the test dataset
df_test = pd.read_csv('fraudTest.csv')

# Drop unnecessary columns in the test set
df_test.drop(columns=["Unnamed: 0", "trans_date_trans_time", "cc_num", "first", "last", "street", "dob", "trans_num"], inplace=True)

# Apply the same unique value limit to the test set
for column in high_cardinality_columns:
    df_test = limit_unique_values(df_test, column)

# Retain the 'is_fraud' column before one-hot encoding
y_test = df_test['is_fraud']

# One-hot encode the test data
df_test_processed = pd.get_dummies(df_test.drop(columns='is_fraud', axis=1))

# Ensure the training and test set have the same dummy variables
df_test_processed = df_test_processed.reindex(columns=X_train.columns, fill_value=0)

# Split features and target variable for the test set
X_test = df_test_processed

# Initialize models
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Decision Tree Classifier': DecisionTreeClassifier(),
    'Random Forest Classifier': RandomForestClassifier(n_estimators=100),
}

# Train and evaluate models
for name, model in models.items():
    print(f"--- {name} ---")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print(f"Accuracy: {accuracy:.2f}")
    print(f"F1 Score: {f1:.2f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print("\n")
