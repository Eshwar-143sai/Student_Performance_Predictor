import os
import joblib
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from data_preprocessing import load_and_preprocess_data

def evaluate_model(true, predicted):
    rmse = np.sqrt(mean_squared_error(true, predicted))
    r2 = r2_score(true, predicted)
    return rmse, r2

def main():
    filepath = 'StudentsPerformance.csv'
    if not os.path.exists(filepath):
        print(f"Dataset '{filepath}' not found. Please place it in the project root or run generate_mock_data.py")
        return

    print(f"Loading and preprocessing data from {filepath}...\n")
    X_train, X_test, y_train, y_test = load_and_preprocess_data(filepath)
    
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42)
    }
    
    os.makedirs('models', exist_ok=True)
    
    print("\n--- Training Models ---")
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        
        train_rmse, train_r2 = evaluate_model(y_train, y_train_pred)
        test_rmse, test_r2 = evaluate_model(y_test, y_test_pred)
        
        print(f"{name} Performance:")
        print(f"  Train: RMSE = {train_rmse:.4f}, R2 = {train_r2:.4f}")
        print(f"  Test:  RMSE = {test_rmse:.4f}, R2 = {test_r2:.4f}")
        
        # Save model
        model_filename = f"models/{name.replace(' ', '_').lower()}.joblib"
        joblib.dump(model, model_filename)
        print(f"Saved model to {model_filename}")

if __name__ == "__main__":
    main()
