# Student Math Score Predictor

This is a complete machine learning project for predicting student math scores based on a dataset like `StudentsPerformance.csv`.

## Setup

1. Create a virtual environment and install the requirements:
   ```bash
   pip install -r requirements.txt
   ```

2. Make sure you have `StudentsPerformance.csv` in the root directory. If you don't have it, you can generate a mock dataset:
   ```bash
   python generate_mock_data.py
   ```

## Project Structure

- `data_preprocessing.py`: Handles data exploration, cleaning, scaling, encoding, and creating the `preprocessor.joblib`. It will save a correlation matrix in `plots/`.
- `train_models.py`: Uses the preprocessed data to train Linear Regression, Random Forest, and Gradient Boosting models. It evaluates using RMSE and R^2 and saves the models in the `models/` folder.
- `predict.py`: Loads the saved `preprocessor` and `models` to predict a new student's math score based on specified features.
- `.gitignore`: ignores data sets, models, and plot outputs.

## How to Run

**1. Train the Models & Explore Data**
Run the training script (which automatically executes the preprocessing script):
```bash
python train_models.py
```

**2. Make Predictions**
After models are trained, use the predict script:
```bash
python predict.py
```
