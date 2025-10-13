import json
import os
import subprocess
import sys


def test_train_produces_artifacts(tmp_path):
    here = os.path.dirname(os.path.dirname(__file__))
    train_script = os.path.join(here, "src", "train.py")

    proc = subprocess.run(
        [sys.executable, train_script], cwd=tmp_path, capture_output=True, text=True
    )
    assert proc.returncode == 0, proc.stderr

    model_path = tmp_path / "model.joblib"
    metrics_path = tmp_path / "metrics.json"
    assert model_path.exists()
    assert metrics_path.exists()

    metrics = json.loads(metrics_path.read_text())
    assert "rmse" in metrics
    assert metrics["rmse"] < 70.0
