import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Load dataset
def load_data():
    # Using a sample dataset from sklearn for demonstration purposes
    from sklearn.datasets import load_iris
    data = load_iris()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['target'] = data.target
    return df

data = load_data()

# Exploratory Data Analysis (EDA)
def eda(data):
    print("\nDataset Overview:\n")
    print(data.head())
    print("\nDataset Information:\n")
    print(data.info())
    print("\nClass Distribution:\n")
    print(data['target'].value_counts())

eda(data)

# Feature Selection
X = data.iloc[:, :-1]
y = data['target']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Data Preprocessing
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Model Training
classifier = RandomForestClassifier(n_estimators=100, random_state=42)
classifier.fit(X_train, y_train)

# Predictions
y_pred = classifier.predict(X_test)

# Evaluation
def evaluate_model(y_test, y_pred):
    print("\nConfusion Matrix:\n")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred))
    print("\nAccuracy Score:", accuracy_score(y_test, y_pred))

evaluate_model(y_test, y_pred)

# Feature Importance
importances = classifier.feature_importances_
feature_names = X.columns

print("\nFeature Importances:\n")
for name, importance in zip(feature_names, importances):
    print(f"{name}: {importance:.4f}")

# Save Model
import joblib
joblib.dump(classifier, 'random_forest_model.pkl')

print("\nModel saved as 'random_forest_model.pkl'")
