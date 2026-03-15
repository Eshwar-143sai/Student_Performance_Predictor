import pandas as pd
import joblib
import os
import sys

def predict_student_grade(student_data_dict, model_name="best_model"):
    """
    Predict final grade (G3) for a single student.
    student_data_dict should be a dictionary with all input features.
    """
    preprocessor_path = 'models/preprocessor.joblib'
    
    if model_name == "best_model":
        model_path = 'models/best_model.pkl'
    else:
        model_path = f"models/{model_name}.joblib"
    
    if not os.path.exists(preprocessor_path) or not os.path.exists(model_path):
        print(f"Error: Could not find preprocessor or model ('{model_name}').")
        print("Please run `python train_models.py` first to generate them.")
        sys.exit(1)
        
    preprocessor = joblib.load(preprocessor_path)
    model = joblib.load(model_path)
            
    df = pd.DataFrame([student_data_dict])
    X_processed = preprocessor.transform(df)
    prediction = model.predict(X_processed)
    
    return prediction[0]

if __name__ == "__main__":
    print("--- Student Final Grade (G3) Prediction Tool ---")
    
    # Example student data matching student_data.csv features
    sample_student = {
        'school': 'GP',
        'sex': 'F',
        'age': 18,
        'address': 'U',
        'famsize': 'GT3',
        'Pstatus': 'A',
        'Medu': 4,
        'Fedu': 4,
        'Mjob': 'at_home',
        'Fjob': 'teacher',
        'reason': 'course',
        'guardian': 'mother',
        'traveltime': 2,
        'studytime': 2,
        'failures': 0,
        'schoolsup': 'yes',
        'famsup': 'no',
        'paid': 'no',
        'activities': 'no',
        'nursery': 'yes',
        'higher': 'yes',
        'internet': 'no',
        'romantic': 'no',
        'famrel': 4,
        'freetime': 3,
        'goout': 4,
        'Dalc': 1,
        'Walc': 1,
        'health': 3,
        'absences': 6,
        'G1': 5,
        'G2': 6
    }
    
    print("\nPredicting final grade for sample student:")
    for k, v in sample_student.items():
        print(f"  {k}: {v}")
        
    try:
        score_lr = predict_student_grade(sample_student, "linear_regression")
        score_rf = predict_student_grade(sample_student, "random_forest")
        score_gb = predict_student_grade(sample_student, "gradient_boosting")
        score_best = predict_student_grade(sample_student, "best_model")
        
        print("\n--- Predictions (Grade out of 20) ---")
        print(f"Linear Regression:  {score_lr:.2f}")
        print(f"Random Forest:      {score_rf:.2f}")
        print(f"Gradient Boosting:  {score_gb:.2f}")
        print(f"Best Model:         {score_best:.2f}")
    except Exception as e:
        print(f"\nError occurred during prediction: {e}")
