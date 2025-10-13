import json
import os
import random
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def set_seeds(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = "0"


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def main() -> None:
    set_seeds(42)

    # Load data
    data = load_diabetes(as_frame=True)
    df: pd.DataFrame = data.frame
    X = df.drop(columns=["target"])
    y = df["target"]

    # Hold-out split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Candidate models
    ridge = Pipeline(
        [("scaler", StandardScaler()), ("model", Ridge(alpha=1.0, random_state=42))]
    )
    rf = Pipeline(
        [
            (
                "model",
                RandomForestRegressor(
                    n_estimators=400,
                    max_depth=None,
                    min_samples_leaf=1,
                    random_state=42,
                    n_jobs=-1,
                ),
            )
        ]
    )

    candidates = [("Ridge", ridge), ("RandomForestRegressor", rf)]
    results = []

    best_name = None
    best_model = None
    best_rmse = float("inf")

    for name, pipe in candidates:
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)
        model_rmse = rmse(y_test, preds)
        results.append({"model_type": name, "rmse": model_rmse})
        if model_rmse < best_rmse:
            best_rmse = model_rmse
            best_model = pipe
            best_name = name

    # OPTIONAL: derive a simple "high-risk" flag to show extra metrics
    # Threshold = 75th percentile of y_train (top quartile as high risk)
    thresh = float(np.quantile(y_train, 0.75))
    y_true_flag = (y_test > thresh).astype(int)
    y_pred_flag = (best_model.predict(X_test) > thresh).astype(int)
    prec = float(precision_score(y_true_flag, y_pred_flag, zero_division=0))
    rec = float(recall_score(y_true_flag, y_pred_flag, zero_division=0))

    # Save the chosen model
    # After computing: best_name, best_model, best_rmse, prec, rec, thresh ...

    joblib.dump(best_model, "model.joblib")

    metrics = {
        "version": "v0.2",
        "random_state": 42,
        "train_time_utc": datetime.utcnow().isoformat() + "Z",
        "sklearn_version": "1.3.2",
        "candidates": results,
        "best": {"model_type": best_name, "rmse": best_rmse},
        # add top-level rmse for backward compatibility with tests
        "rmse": best_rmse,
        "risk_threshold": {"type": "percentile", "value": 0.75, "y_threshold": thresh},
        "classification_at_threshold": {"precision": prec, "recall": rec},
    }

    with open("metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)


    print(
        f"[v0.2] Saved model.joblib using {best_name} | RMSE={best_rmse:.4f} | "
        f"precision@thr={prec:.3f} recall@thr={rec:.3f}"
    )


if __name__ == "__main__":
    main()
