from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import pandas as pd
import os

app = FastAPI(title="Student Math Score Predictor API")

# Load models at startup
preprocessor_path = 'models/preprocessor.joblib'
best_model_path = 'models/best_model.pkl'

preprocessor = None
model = None

@app.on_event("startup")
def load_models():
    global preprocessor, model
    if os.path.exists(preprocessor_path) and os.path.exists(best_model_path):
        preprocessor = joblib.load(preprocessor_path)
        model = joblib.load(best_model_path)
        print("Models loaded successfully.")
    else:
        print("Warning: Models not found. Please run train_models.py first.")

class StudentData(BaseModel):
    gender: str = Field(..., alias="gender")
    race_ethnicity: str = Field(..., alias="race/ethnicity")
    parental_level_of_education: str = Field(..., alias="parental level of education")
    lunch: str = Field(..., alias="lunch")
    test_preparation_course: str = Field(..., alias="test preparation course")
    reading_score: int = Field(..., alias="reading score")
    writing_score: int = Field(..., alias="writing score")

    model_config = {
        "populate_by_name": True
    }

@app.post("/predict")
def predict(student: StudentData):
    if preprocessor is None or model is None:
        raise HTTPException(status_code=500, detail="Models are not loaded. Train the models first.")
        
    # Convert input to DataFrame using aliases as columns so preprocessor matches
    student_dict = student.model_dump(by_alias=True)
    df = pd.DataFrame([student_dict])
    
    try:
        X_processed = preprocessor.transform(df)
        prediction = model.predict(X_processed)
        return {"predicted_math_score": float(prediction[0])}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
