## [v0.2] – 2025-10-16
**Change:** Model selection between **Ridge** and **RandomForestRegressor**; choose best by held-out RMSE.  
**API:** No change — `/predict` still returns a continuous risk score.  
**Artifacts:** See `metrics.json` attached to the v0.2 GitHub Release.

**Metrics (held-out test; random_state=42; sklearn=1.3.2; pandas=2.2.2)**
- v0.1 RMSE: **53.8534**
- v0.2 RMSE (best): **53.7775**
- **Δ absolute:** −0.0759  
- **Δ percent:** −0.14%
- Optional classification at 75th percentile (for context only):
  - precision = **1.000**
  - recall = **0.300**

**Ops signals**
- Image tag: `ghcr.io/gustav951/virtual-diabetes-clinic-grupp-g:v0.2`
- Startup: unchanged vs v0.1 (Uvicorn prints “Application startup complete” in ~X s on GitHub runner)
- Image size: ~X MB (≈ v0.1)

**Rationale**
- **Ridge** adds L2 regularization (reduces variance vs OLS).
- **RandomForest** captures non-linearities and interactions.
- We pick the **lower RMSE** of the two on a fixed split for deterministic improvement.

---

## [v0.1] – 2025-10-15
**Baseline:** `StandardScaler + LinearRegression`, FastAPI service with `/health` and `/predict`.  
**Artifacts:** `model.joblib`, `metrics.json`.  
**Image:** `ghcr.io/gustav951/virtual-diabetes-clinic-grupp-g:v0.1`  
**CI/CD:** Push/PR CI (lint, tests, smoke), tag Release → build, smoke, push to GHCR, publish Release.

**Metrics (held-out test; random_state=42; sklearn=1.3.2)**
- v0.1 RMSE: **53.8534**
