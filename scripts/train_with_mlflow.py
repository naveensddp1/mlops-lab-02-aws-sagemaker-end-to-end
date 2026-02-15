# scripts/train_with_mlflow.py

import os
import json
import joblib
import pandas as pd
import xgboost as xgb
import mlflow
import mlflow.xgboost

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# SageMaker channels
TRAIN_DIR = os.environ.get("SM_CHANNEL_TRAIN", "/opt/ml/input/data/train")
MODEL_DIR = os.environ.get("SM_MODEL_DIR", "/opt/ml/model")

# MLflow
MLFLOW_URI = os.environ["MLFLOW_TRACKING_URI"].rstrip("/")
mlflow.set_tracking_uri(MLFLOW_URI)
mlflow.set_experiment("wine-quality-training")

# Load CSV from train channel (expects exactly one CSV in the folder)
csv_files = [f for f in os.listdir(TRAIN_DIR) if f.endswith(".csv")]
if len(csv_files) == 0:
    raise RuntimeError(f"No CSV found in {TRAIN_DIR}")
csv_path = os.path.join(TRAIN_DIR, csv_files[0])

df = pd.read_csv(csv_path)

# Label is last column in your wine.csv
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

params = {
    "objective": "reg:squarederror",
    "max_depth": 5,
    "eta": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "n_estimators": 300,
    "random_state": 42,
}

with mlflow.start_run():
    mlflow.log_params(params)

    model = xgb.XGBRegressor(**params)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    rmse = mean_squared_error(y_test, preds, squared=False)
    r2 = r2_score(y_test, preds)

    mlflow.log_metric("rmse", float(rmse))
    mlflow.log_metric("r2", float(r2))

    # Log model to MLflow
    mlflow.xgboost.log_model(model, artifact_path="model")

    # Save model for SageMaker hosting
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, os.path.join(MODEL_DIR, "model.joblib"))

    # Save metrics artifact too
    metrics_path = os.path.join(MODEL_DIR, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump({"rmse": float(rmse), "r2": float(r2)}, f)
    mlflow.log_artifact(metrics_path)

print("Training complete. Model saved to:", MODEL_DIR)
