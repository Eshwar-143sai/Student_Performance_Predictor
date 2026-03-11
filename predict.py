import pandas as pd
import joblib
import os
import sys

def predict_student_score(student_data_dict, model_name="random_forest"):
    """
    Predict math score for a single student.
    student_data_dict should be a dictionary with the features.
    """
    preprocessor_path = 'models/preprocessor.joblib'
    model_path = f"models/{model_name}.joblib"
    
    if not os.path.exists(preprocessor_path) or not os.path.exists(model_path):
        print(f"Error: Could not find preprocessor or model ('{model_name}').")
        print("Please run `python train_models.py` first to generate them.")
        sys.exit(1)
        
    # Load preprocessor and model
    preprocessor = joblib.load(preprocessor_path)
    model = joblib.load(model_path)
            
    # Convert dict to DataFrame
    df = pd.DataFrame([student_data_dict])
    
    # Process features
    X_processed = preprocessor.transform(df)
    
    # Predict
    prediction = model.predict(X_processed)
    
    return prediction[0]

if __name__ == "__main__":
    print("--- Math Score Prediction Tool ---")
    
    # Example student data
    sample_student = {
        'gender': 'female',
        'race/ethnicity': 'group B',
        'parental level of education': "bachelor's degree",
        'lunch': 'standard',
        'test preparation course': 'none',
        'reading score': 72,
        'writing score': 74
    }
    
    print("\nPredicting math score for sample student:")
    for k, v in sample_student.items():
        print(f"  {k}: {v}")
        
    try:
        # Predict using random forest by default
        score_rf = predict_student_score(sample_student, "random_forest")
        score_lr = predict_student_score(sample_student, "linear_regression")
        score_gb = predict_student_score(sample_student, "gradient_boosting")
        
        print("\n--- Predictions ---")
        print(f"Linear Regression Predicted Score: {score_lr:.2f}")
        print(f"Random Forest Predicted Score:     {score_rf:.2f}")
        print(f"Gradient Boosting Predicted Score: {score_gb:.2f}")
    except Exception as e:
        print(f"\nError occurred during prediction: {e}")
