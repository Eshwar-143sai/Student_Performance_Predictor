import os
import joblib
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from data_preprocessing import load_and_preprocess_data

def evaluate_model(true, predicted):
    rmse = np.sqrt(mean_squared_error(true, predicted))
    r2 = r2_score(true, predicted)
    return rmse, r2

def main():
    filepath = 'student_data.csv'
    if not os.path.exists(filepath):
        print(f"Dataset '{filepath}' not found. Please place it in the project root.")
        return

    print(f"Loading and preprocessing data from {filepath}...\n")
    X_train, X_test, y_train, y_test = load_and_preprocess_data(filepath)
    
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(random_state=42)
    }

    param_grids = {
        "Linear Regression": {},
        "Random Forest": {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5]
        },
        "Gradient Boosting": {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7]
        }
    }
    
    os.makedirs('models', exist_ok=True)
    
    print("\n--- Training Models and Tuning Hyperparameters ---")
    
    best_model_name = ""
    best_model = None
    best_test_r2 = -float("inf")
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        
        if param_grids[name]:
            grid_search = GridSearchCV(
                estimator=model, param_grid=param_grids[name],
                cv=5, scoring='r2', n_jobs=-1, verbose=1
            )
            grid_search.fit(X_train, y_train)
            best_tuned_model = grid_search.best_estimator_
            print(f"Best Parameters for {name}: {grid_search.best_params_}")
            model_to_evaluate = best_tuned_model
        else:
            model.fit(X_train, y_train)
            model_to_evaluate = model
            
        y_train_pred = model_to_evaluate.predict(X_train)
        y_test_pred = model_to_evaluate.predict(X_test)
        
        train_rmse, train_r2 = evaluate_model(y_train, y_train_pred)
        test_rmse, test_r2 = evaluate_model(y_test, y_test_pred)
        
        print(f"{name} Performance:")
        print(f"  Train: RMSE = {train_rmse:.4f}, R2 = {train_r2:.4f}")
        print(f"  Test:  RMSE = {test_rmse:.4f}, R2 = {test_r2:.4f}")
        
        # Save individual model
        model_filename = f"models/{name.replace(' ', '_').lower()}.joblib"
        joblib.dump(model_to_evaluate, model_filename)
        print(f"Saved {name} to {model_filename}")
        
        # Keep track of the best model
        if test_r2 > best_test_r2:
            best_test_r2 = test_r2
            best_model = model_to_evaluate
            best_model_name = name

    print(f"\n--- Best Model Selection ---")
    print(f"The best model overall is {best_model_name} with an R2 score of {best_test_r2:.4f}.")
    
    # Save the single best-performing model
    best_model_path = "models/best_model.pkl"
    joblib.dump(best_model, best_model_path)
    print(f"Saved the best overall model to {best_model_path}")

if __name__ == "__main__":
    main()
