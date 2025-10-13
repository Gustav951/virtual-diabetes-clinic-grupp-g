import json
import os
from typing import Any, Dict, List

import joblib
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ValidationError

MODEL_PATH = os.getenv("MODEL_PATH", "model.joblib")
MODEL_VERSION = os.getenv("MODEL_VERSION", "dev")


class PatientFeatures(BaseModel):
    age: float
    sex: float
    bmi: float
    bp: float
    s1: float
    s2: float
    s3: float
    s4: float
    s5: float
    s6: float


def load_model(path: str):
    try:
        return joblib.load(path)
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {e}") from e


app = FastAPI(title="Virtual Diabetes Clinic Risk API", version=MODEL_VERSION)
model = load_model(MODEL_PATH)


@app.exception_handler(ValidationError)
async def validation_error_handler(request: Request, exc: ValidationError):
    return JSONResponse(
        status_code=422, content={"error": "validation_error", "detail": json.loads(exc.json())}
    )


@app.exception_handler(RequestValidationError)
async def request_validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=422,
        content={"error": "validation_error", "detail": exc.errors()},
    )


@app.exception_handler(Exception)
async def generic_error_handler(request: Request, exc: Exception):
    return JSONResponse(status_code=400, content={"error": "bad_request", "detail": str(exc)})


@app.get("/health")
def health() -> Dict[str, Any]:
    return {"status": "ok", "model_version": MODEL_VERSION}


@app.post("/predict")
def predict(features: PatientFeatures) -> Dict[str, float]:
    try:
        X: List[List[float]] = [
            [
                features.age,
                features.sex,
                features.bmi,
                features.bp,
                features.s1,
                features.s2,
                features.s3,
                features.s4,
                features.s5,
                features.s6,
            ]
        ]
        prediction = float(model.predict(X)[0])
        return {"prediction": prediction}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
