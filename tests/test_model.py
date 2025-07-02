from src.pipline import create_features
import pytest
import joblib
import pandas as pd
from pathlib import Path
import xgboost
import sys
sys.path.append(str(Path(__file__).parent.parent))

MODEL_DIR = Path(__file__).parent.parent / "src" / "model"


@pytest.fixture(scope="module")
def pipeline():
    return joblib.load(MODEL_DIR / "pipeline.pkl")


@pytest.fixture(scope="module")
def model():
    return joblib.load(MODEL_DIR / "model.pkl")


def test_model_loading(model):
    assert hasattr(model, "predict") and callable(model.predict)


def test_prediction_shape(pipeline, model):
    """
    Run the two provided real rows through pipeline -> model and
    assert we get back exactly two predictions.
    """
    raw = [
        {
            "TransactionId": "TransactionId_16559",
            "BatchId": "BatchId_79110",
            "AccountId": "AccountId_571",
            "SubscriptionId": "SubscriptionId_873",
            "CustomerId": "CustomerId_908",
            "CurrencyCode": "UGX",
            "CountryCode": 256,
            "ProviderId": "ProviderId_5",
            "ProductId": "ProductId_15",
            "ProductCategory": "financial_services",
            "ChannelId": "ChannelId_3",
            "Amount": 2200.0,
            "Value": 2200.0,
            "TransactionStartTime": "2018-11-15T05:54:12Z",
            "PricingStrategy": 2,
            "FraudResult": 0
        },
        {
            "TransactionId": "TransactionId_79455",
            "BatchId": "BatchId_122807",
            "AccountId": "AccountId_571",
            "SubscriptionId": "SubscriptionId_873",
            "CustomerId": "CustomerId_908",
            "CurrencyCode": "UGX",
            "CountryCode": 256,
            "ProviderId": "ProviderId_5",
            "ProductId": "ProductId_15",
            "ProductCategory": "financial_services",
            "ChannelId": "ChannelId_3",
            "Amount": 2200.0,
            "Value": 2200.0,
            "TransactionStartTime": "2018-11-15T05:55:12Z",
            "PricingStrategy": 2,
            "FraudResult": 0
        }
    ]
    df = pd.DataFrame(raw)
    X = pipeline.transform(df)
    preds = model.predict(X)
    assert preds.shape[0] == df.shape[0], f"Expected {
        df.shape[0]} preds, got {
        preds.shape[0]}"
