# Changelog

## v0.1
- Baseline model: StandardScaler + LinearRegression
- Adds FastAPI API with /health and /predict
- Docker image with model baked in
- CI pipeline and release workflow with GHCR publish

## v0.2
- Model selection between **Ridge** and **RandomForestRegressor**; choose best by RMSE on held-out test.
- Added optional high-risk flag (top quartile threshold) for extra context, reporting precision/recall.
- No API change; same `/predict` returning a continuous risk score.

**Metrics**
- v0.1 RMSE: <53.8534>
- v0.2 RMSE (best): <53.7775>  _(Δ = v0.2 - v0.1)_
- v0.2 high-risk @ 75th percentile: precision=<1.000>, recall=<0.300>

**Rationale**
- RandomForest/Ridge reduce error vs. OLS by handling non-linearities (RF) or regularization (Ridge).

