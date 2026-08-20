from fastapi import FastAPI
from pydantic import BaseModel

from src.predict import load_model, predict_transaction


MODEL_PATH = "models/model.joblib"

app = FastAPI(
    title="Digital Payment Fraud Detection API",
    version="1.0.0"
)

artifact = load_model(MODEL_PATH)


class Transaction(BaseModel):
    transaction_amount: float
    transaction_type: str
    payment_mode: str
    device_type: str
    device_location: str
    account_age_days: int
    transaction_hour: int
    previous_failed_attempts: int
    avg_transaction_amount: float
    is_international: int
    ip_risk_score: float
    login_attempts_last_24h: int


@app.get("/")
def root():
    return {"message": "Digital Payment Fraud Detection API"}


@app.post("/predict")
def predict(transaction: Transaction):
    prediction = predict_transaction(
        transaction.model_dump(),
        artifact
    )

    return {
        "fraud_prediction": prediction
    }