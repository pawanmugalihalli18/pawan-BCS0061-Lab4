from fastapi import FastAPI
import joblib
import pandas as pd

app = FastAPI()

# Load model
model = joblib.load("model.pkl")

@app.get("/")
def home():
    return {"message": "Wine Quality API is running"}

@app.post("/predict")
def predict(data: dict):
    try:
        df = pd.DataFrame([data])
        prediction = model.predict(df)[0]

        return {
            "name": "Pawan",
            "roll_no": "2022bcs0061",
            "wine_quality": float(prediction)
        }

    except Exception as e:
        return {"error": str(e)}