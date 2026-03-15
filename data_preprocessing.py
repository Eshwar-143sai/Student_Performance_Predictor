import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import joblib
import os

TARGET_COL = 'G3'

CATEGORICAL_FEATURES = [
    'school', 'sex', 'address', 'famsize', 'Pstatus',
    'Mjob', 'Fjob', 'reason', 'guardian',
    'schoolsup', 'famsup', 'paid', 'activities',
    'nursery', 'higher', 'internet', 'romantic'
]

NUMERIC_FEATURES = [
    'age', 'Medu', 'Fedu', 'traveltime', 'studytime',
    'failures', 'famrel', 'freetime', 'goout',
    'Dalc', 'Walc', 'health', 'absences', 'G1', 'G2'
]

def explore_data(df):
    print("--- Dataset Info ---")
    print(df.info())
    print("\n--- Summary Statistics ---")
    print(df.describe())
    print("\n--- Missing Values ---")
    print(df.isnull().sum())
    
    # Save a correlation plot for numeric variables
    os.makedirs('plots', exist_ok=True)
    plt.figure(figsize=(12, 10))
    numeric_df = df.select_dtypes(include=[np.number])
    sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt=".2f", 
                linewidths=0.5, square=True)
    plt.title("Correlation Matrix — Student Performance Dataset")
    plt.tight_layout()
    plt.savefig('plots/correlation_matrix.png', dpi=150)
    plt.close()
    print("Correlation matrix saved to plots/correlation_matrix.png")

def load_and_preprocess_data(filepath, target_col=TARGET_COL, save_pipeline=True):
    df = pd.read_csv(filepath)
    explore_data(df)
    
    # Separate features and target
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Use predefined feature lists for consistency
    num_features = [f for f in NUMERIC_FEATURES if f in X.columns]
    cat_features = [f for f in CATEGORICAL_FEATURES if f in X.columns]
    
    # Build preprocessor
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, num_features),
            ('cat', categorical_transformer, cat_features)
        ])
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Fit and transform
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    if save_pipeline:
        os.makedirs('models', exist_ok=True)
        joblib.dump(preprocessor, 'models/preprocessor.joblib')
        print("Preprocessing pipeline saved to models/preprocessor.joblib")
        
    return X_train_processed, X_test_processed, y_train, y_test

if __name__ == "__main__":
    load_and_preprocess_data("student_data.csv")
