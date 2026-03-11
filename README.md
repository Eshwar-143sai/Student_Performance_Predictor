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

**2. Make Predictions (CLI)**
After models are trained, use the predict script:
```bash
python predict.py
```

**3. Run the Prediction API (FastAPI)**
We have a FastAPI application with a `/predict` endpoint that uses the single best tuned model. The models were evaluated using 5-fold cross-validation with the following test set performance:

- **Linear Regression**: RMSE = 13.9318, R2 = 0.8804
- **Random Forest (Tuned)**: RMSE = 14.1678, R2 = 0.8763
- **Gradient Boosting (Tuned)**: RMSE = 13.5614, R2 = 0.8867

*The best overall model was Gradient Boosting with an R2 of 0.8867.*

To run the API:

```bash
uvicorn app:app --reload
```

Then you can send a POST request with JSON data to `http://localhost:8000/predict`.

**API Usage with `curl`:**
```bash
curl -X 'POST' \
  'http://localhost:8000/predict' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "gender": "female",
  "race/ethnicity": "group B",
  "parental level of education": "bachelor'\''s degree",
  "lunch": "standard",
  "test preparation course": "none",
  "reading score": 72,
  "writing score": 74
}'
```

You can also visit `http://localhost:8000/docs` to test the API directly using Swagger UI.
