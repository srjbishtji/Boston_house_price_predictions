import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
import os

def test_prediction():
    column_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
    df = pd.read_csv('data/housing.csv', header=None, names=column_names, sep=r'\s+')
    X = df.drop('MEDV', axis=1)
    y = df['MEDV']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model_path = 'models/linear_regression_model.pkl'
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}. Ensure the model is trained and saved.")
    
    model = joblib.load(model_path)
    y_pred = model.predict(X_test)
    
    assert len(y_pred) == len(X_test), f"Prediction length mismatch! Expected {len(X_test)}, got {len(y_pred)}"
    print("Model test passed!")

if __name__ == "__main__":
    test_prediction()
