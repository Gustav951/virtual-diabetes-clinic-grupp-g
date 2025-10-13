# Virtual Diabetes Clinic Risk Service

Predicts short term progression using scikit-learn diabetes dataset. Returns a continuous risk score.

## Local usage
```bash
pip install -r requirements.txt
python src/train.py
uvicorn src.predict_service:app --host 0.0.0.0 --port 8000

### v0.2
- Improved model (Ridge vs RandomForest) with lower RMSE.
- Pull: `docker pull ghcr.io/gustav951/virtual-diabetes-clinic-grupp-g:v0.2`

