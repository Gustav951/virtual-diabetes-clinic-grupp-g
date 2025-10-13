import json
import os
import random
from datetime import datetime

import joblib
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def set_seeds(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = "0"


def main() -> None:
    set_seeds(42)

    data = load_diabetes(as_frame=True)
    X = data.frame.drop(columns=["target"])
    y = data.frame["target"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    pipe = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("model", LinearRegression()),
        ]
    )
    pipe.fit(X_train, y_train)

    preds = pipe.predict(X_test)
    rmse = float(np.sqrt(mean_squared_error(y_test, preds)))

    joblib.dump(pipe, "model.joblib")

    meta = {
        "version": "v0.1",
        "model_type": "LinearRegression",
        "random_state": 42,
        "train_time_utc": datetime.utcnow().isoformat() + "Z",
        "sklearn_version": "1.3.2",
        "rmse": rmse,
    }
    with open("metrics.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved model.joblib, RMSE={rmse:.4f}")


if __name__ == "__main__":
    main()
