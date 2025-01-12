from flask import Flask, request, jsonify
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import os

# Flask app
app = Flask(__name__)

# Helper function to load dataset
def load_data():
    from sklearn.datasets import load_iris
    data = load_iris()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['target'] = data.target
    return df

# Helper function to train and save the model
def train_and_save_model():
    print("Training the model...")
    data = load_data()
    X = data.iloc[:, :-1]
    y = data['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Data Preprocessing
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Train Model
    classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    classifier.fit(X_train, y_train)

    # Save Model and Scaler
    joblib.dump(classifier, 'random_forest_model.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    print("Model and scaler saved successfully!")
    return data, X, y, X_train, X_test, y_train, y_test

# Check if the model file exists, if not, train and save it
if not os.path.exists('random_forest_model.pkl') or not os.path.exists('scaler.pkl'):
    data, X, y, X_train, X_test, y_train, y_test = train_and_save_model()
else:
    print("Loading existing model and scaler...")
    data = load_data()
    X = data.iloc[:, :-1]
    y = data['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    scaler = joblib.load('scaler.pkl')

# API Endpoints
@app.route('/eda', methods=['GET'])
def eda():
    """Endpoint for Exploratory Data Analysis"""
    class_distribution = data['target'].value_counts().to_dict()
    return jsonify({
        "overview": data.head().to_dict(),
        "class_distribution": class_distribution
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Endpoint for predictions"""
    model = joblib.load('random_forest_model.pkl')
    scaler = joblib.load('scaler.pkl')
    input_data = request.json.get("features")  # Input data as JSON
    if not input_data:
        return jsonify({"error": "No input data provided"}), 400
    input_data = pd.DataFrame([input_data], columns=X.columns)
    scaled_data = scaler.transform(input_data)
    prediction = model.predict(scaled_data)
    return jsonify({"prediction": int(prediction[0])})

@app.route('/evaluate', methods=['GET'])
def evaluate():
    """Endpoint for model evaluation"""
    y_pred = joblib.load('random_forest_model.pkl').predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    matrix = confusion_matrix(y_test, y_pred).tolist()
    accuracy = accuracy_score(y_test, y_pred)
    return jsonify({
        "confusion_matrix": matrix,
        "classification_report": report,
        "accuracy_score": accuracy
    })

@app.route('/feature_importance', methods=['GET'])
def feature_importance():
    """Endpoint for feature importance"""
    model = joblib.load('random_forest_model.pkl')
    importances = model.feature_importances_
    feature_names = X.columns
    importance_dict = {name: importance for name, importance in zip(feature_names, importances)}
    return jsonify(importance_dict)

# Run Flask app
if __name__ == '__main__':
    app.run(debug=True)
