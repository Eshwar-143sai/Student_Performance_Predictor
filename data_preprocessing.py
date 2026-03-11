import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import joblib
import os

def explore_data(df):
    print("--- Dataset Info ---")
    print(df.info())
    print("\n--- Summary Statistics ---")
    print(df.describe())
    print("\n--- Missing Values ---")
    print(df.isnull().sum())
    
    # Save a basic correlation plot for target variables
    os.makedirs('plots', exist_ok=True)
    plt.figure(figsize=(8, 6))
    numeric_df = df.select_dtypes(include=[np.number])
    sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Correlation Matrix")
    plt.tight_layout()
    plt.savefig('plots/correlation_matrix.png')
    plt.close()

def load_and_preprocess_data(filepath, target_col='math score', save_pipeline=True):
    df = pd.read_csv(filepath)
    explore_data(df)
    
    # Separate features and target
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Identify numeric and categorical columns
    num_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    cat_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Build preprocessor
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, num_features),
            ('cat', categorical_transformer, cat_features)
        ])
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Fit and transform
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    if save_pipeline:
        os.makedirs('models', exist_ok=True)
        joblib.dump(preprocessor, 'models/preprocessor.joblib')
        print("Preprocessing pipeline saved to models/preprocessor.joblib")
        
    return X_train_processed, X_test_processed, y_train, y_test

if __name__ == "__main__":
    load_and_preprocess_data("StudentsPerformance.csv")
