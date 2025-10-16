Virtual Diabetes Clinic Triage — Risk API
=================================================

Predicts short-term diabetes progression as a continuous risk score (higher ≈ worse).
Intended for a triage dashboard that sorts patients by predicted progression.

- API: FastAPI (/health, /predict)
- Model:
  - v0.1 — StandardScaler + LinearRegression
  - v0.2 — model selection between Ridge and RandomForestRegressor, choose best by RMSE;
           optional high-risk flag metrics (precision/recall)
- Delivery: Docker image with model baked in (self-contained), published to GHCR via GitHub Actions

-------------------------------------------------
Quick start (Docker — recommended)
-------------------------------------------------

Pull a published image and run it (port 8000):

# Baseline
docker pull ghcr.io/gustav951/virtual-diabetes-clinic-grupp-g:v0.1
docker run --rm -p 8000:8000 ghcr.io/gustav951/virtual-diabetes-clinic-grupp-g:v0.1

# Improved model
docker pull ghcr.io/gustav951/virtual-diabetes-clinic-grupp-g:v0.2
docker run --rm -p 8000:8000 ghcr.io/gustav951/virtual-diabetes-clinic-grupp-g:v0.2

Smoke test:

curl http://localhost:8000/health
# → {"status":"ok","model_version":"v0.x"}

curl -s -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d '{"age":0.02,"sex":-0.044,"bmi":0.06,"bp":-0.03,"s1":-0.02,"s2":0.03,"s3":-0.02,"s4":0.02,"s5":0.02,"s6":-0.001}'
# → {"prediction": <float>}

Open Swagger UI: http://localhost:8000/docs

If port 8000 is busy, map a different host port, e.g. -p 8080:8000, then use http://localhost:8080.

-------------------------------------------------
Endpoints
-------------------------------------------------

GET /health
Returns:
{"status":"ok","model_version":"v0.x"}

POST /predict
Request JSON (feature names must match scikit-learn diabetes dataset):
{
  "age": 0.02,
  "sex": -0.044,
  "bmi": 0.06,
  "bp": -0.03,
  "s1": -0.02,
  "s2": 0.03,
  "s3": -0.02,
  "s4": 0.02,
  "s5": 0.02,
  "s6": -0.001
}
Response JSON:
{"prediction": 153.25109384}

Bad input (observability requirement):
{"error":"validation_error","detail":[ ... ]}

-------------------------------------------------
Run from source (local dev)
-------------------------------------------------

# 1) Create & activate venv (optional but recommended)
python -m venv .venv
# Windows (Git Bash)
source .venv/Scripts/activate
# macOS/Linux
# source .venv/bin/activate

# 2) Install dependencies
pip install -r requirements.txt

# 3) Train — writes model.joblib + metrics.json to repo root
python src/train.py

# 4) Run API
# macOS/Linux:
uvicorn src.predict_service:app --host 0.0.0.0 --port 8000
# Windows-friendly:
python -m uvicorn src.predict_service:app --host 0.0.0.0 --port 8000

-------------------------------------------------
Tests
-------------------------------------------------

python -m pytest -q

-------------------------------------------------
Building Docker image locally (optional)
-------------------------------------------------

# ensure model artifacts exist in repo root
python src/train.py

# build image (set version label)
docker build --build-arg VERSION=v0.2 -t diabetes-risk:v0.2 .

# run
docker run --rm -p 8000:8000 diabetes-risk:v0.2

-------------------------------------------------
Versions & changes
-------------------------------------------------

v0.1
- Model: StandardScaler + LinearRegression
- Artifacts: model.joblib, metrics.json (includes rmse)
- Image: ghcr.io/gustav951/virtual-diabetes-clinic-grupp-g:v0.1

v0.2
- Model: selection between Ridge and RandomForestRegressor; best by held-out RMSE
- Adds a simple “high-risk” threshold at 75th percentile of y for precision/recall (informative only)
- metrics.json includes candidate metrics, best, top-level rmse (for backward compatibility)
- Image: ghcr.io/gustav951/virtual-diabetes-clinic-grupp-g:v0.2

See CHANGELOG.md for side-by-side metrics and rationale.

-------------------------------------------------
Reproducibility
-------------------------------------------------

- Random seeds set (random_state=42, NumPy seed; PYTHONHASHSEED=0)
- Environment pinned in requirements.txt
- Training script writes metrics.json with RMSE (and v0.2 extras), and model.joblib deterministically
- Docker image bakes the trained model for self-contained runtime

-------------------------------------------------
CI / CD (GitHub Actions)
-------------------------------------------------

CI (on push/PR)
- Lint (ruff, black --check), unit tests (pytest), quick training smoke, upload artifacts (model.joblib, metrics.json).

Release (on tag v*)
- Train model, log in to GHCR, build Docker image (MODEL_VERSION = tag),
  container smoke test, push to GHCR, publish GitHub Release with metrics.json and release notes.

Ensure repo setting: Settings → Actions → General → Workflow permissions → Read and write.

-------------------------------------------------
Collaboration / PR Guidelines
-------------------------------------------------

- Branch naming: feature/<short-name> (e.g., feature/ridge-model)
- Commits: concise, imperative (“Add Ridge model; log metrics”)
- Open a PR to main including:
  - What changed & why
  - Before/after metrics (copy from metrics.json)
- CI must be green (lint, tests) before merge

-------------------------------------------------
Troubleshooting
-------------------------------------------------

- uvicorn or pytest not found → pip install -r requirements.txt; on Windows use python -m uvicorn / python -m pytest
- / returns 404 → use /health and /predict (API service, not a web page)
- Docker pull denied → make GHCR package Public, or docker login ghcr.io with a PAT (read:packages/write:packages)
- Port in use → map a different host port: -p 8080:8000 then hit http://localhost:8080

-------------------------------------------------
Project structure
-------------------------------------------------

.
├── CHANGELOG.md
├── Dockerfile
├── README.md
├── requirements.txt
├── model.joblib           # baked into the image
├── metrics.json
├── src/
│   ├── __init__.py
│   ├── train.py           # v0.1 baseline, v0.2 model selection
│   └── predict_service.py # FastAPI app
├── tests/
│   ├── test_api.py
│   └── test_train.py
├── scripts/
│   ├── smoke.sh
│   └── extract_release_notes.sh
└── .github/workflows/
    ├── ci.yml
    └── release.yml

-------------------------------------------------
License / Attribution
-------------------------------------------------

Uses the open scikit-learn Diabetes dataset via sklearn.datasets.load_diabetes(as_frame=True) for safe, synthetic modeling of “progression risk”.
