# main.py
from fastapi import FastAPI
from pathlib import Path
import joblib
import pandas as pd
from src.api.pydantic_models import TransactionData

MODEL_DIR = Path(__file__).parent.parent / "model"

app = FastAPI()

# Load once at startup
pipeline = joblib.load(MODEL_DIR/"pipeline.pkl")
model    = joblib.load(MODEL_DIR/"model.pkl")

@app.get("/")
def home():
    return {"msg":"The only working dictory is http://127.0.0.1:8000/predict please check it on http://127.0.0.1:8000/docs"}

@app.post("/predict")
def predict(data: TransactionData):
    df = pd.DataFrame([data.dict()])
    X = pipeline.transform(df)

    preds = model.predict(X)
    return {"prediction": int(preds[0])}
