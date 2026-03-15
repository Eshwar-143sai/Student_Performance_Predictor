# Student Performance Predictor 🎓

A machine learning web application that predicts student final grades (G3) based on various personal, family, and academic factors. Built with FastAPI and scikit-learn.

## Features

- **Grade Prediction**: Predicts a student's final grade (0-20 scale) using trained ML models (Linear Regression, Random Forest, Gradient Boosting)
- **Personalized Suggestions**: Provides actionable improvement tips based on the student's profile (study habits, attendance, etc.)
- **Study Help**: Enter a difficult subject/topic and get YouTube tutorial links + concept explanations
- **Premium Web UI**: Dark-themed, responsive design with smooth animations

## Dataset

Uses the UCI Student Performance dataset (`student_data.csv`) with 33 features including:
- **Personal**: school, sex, age, address
- **Family**: family size, parent's education/jobs, guardian
- **Academic**: study time, failures, school support, grades (G1, G2)
- **Lifestyle**: activities, internet, health, alcohol consumption, absences

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Ensure `student_data.csv` is in the project root.

## How to Run

### 1. Train Models
```bash
python train_models.py
```

### 2. CLI Predictions
```bash
python predict.py
```

### 3. Web Application (FastAPI)
```bash
uvicorn app:app --reload
```

Then open:
- **Web UI**: [http://localhost:8000/](http://localhost:8000/)
- **API Docs**: [http://localhost:8000/docs](http://localhost:8000/docs)

## Project Structure

| File | Description |
|------|-------------|
| `app.py` | FastAPI server with `/predict` and `/study-help` endpoints |
| `data_preprocessing.py` | Data loading, cleaning, scaling, encoding |
| `train_models.py` | Model training with hyperparameter tuning (GridSearchCV) |
| `predict.py` | CLI prediction tool |
| `student_data.csv` | UCI Student Performance dataset |
| `static/index.html` | Premium web UI |
| `requirements.txt` | Python dependencies |

## API Endpoints

### POST `/predict`
Predict a student's final grade with improvement suggestions.

### POST `/study-help`
Get YouTube links and concept explanations for a subject/topic.

## About
Predict student final grades using regression models (Random Forest, Gradient Boosting) with hyperparameter tuning. Deployed as a FastAPI endpoint with a premium web interface.
