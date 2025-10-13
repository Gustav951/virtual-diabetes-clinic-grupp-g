import os

import joblib
import pytest
from fastapi.testclient import TestClient

from src.predict_service import MODEL_PATH, app  # noqa: E402


@pytest.fixture(scope="session", autouse=True)
def ensure_model():
    if not os.path.exists(MODEL_PATH):
        from sklearn.datasets import load_diabetes
        from sklearn.linear_model import LinearRegression
        from sklearn.model_selection import train_test_split
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        data = load_diabetes(as_frame=True)
        X = data.frame.drop(columns=["target"])
        y = data.frame["target"]
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
        pipe = Pipeline([("scaler", StandardScaler()), ("model", LinearRegression())])
        pipe.fit(Xtr, ytr)
        joblib.dump(pipe, MODEL_PATH)


client = TestClient(app)


def test_health_ok():
    r = client.get("/health")
    assert r.status_code == 200
    js = r.json()
    assert js["status"] == "ok"
    assert "model_version" in js


def test_predict_ok():
    payload = {
        "age": 0.02,
        "sex": -0.044,
        "bmi": 0.06,
        "bp": -0.03,
        "s1": -0.02,
        "s2": 0.03,
        "s3": -0.02,
        "s4": 0.02,
        "s5": 0.02,
        "s6": -0.001,
    }
    r = client.post("/predict", json=payload)
    assert r.status_code == 200
    js = r.json()
    assert "prediction" in js
    assert isinstance(js["prediction"], float)


def test_predict_bad_input_json_error():
    r = client.post("/predict", json={"age": "oops"})
    assert r.status_code == 422
    js = r.json()
    assert js["error"] == "validation_error"
