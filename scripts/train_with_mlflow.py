# scripts/train_with_mlflow.py

import pandas as pd
import os
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import xgboost as xgb
import joblib

MLFLOW_URI = os.environ["MLFLOW_TRACKING_URI"]

mlflow.set_tracking_uri(MLFLOW_URI)
mlflow.set_experiment("wine-quality")

input_path = "/opt/ml/input/data/train/wine.csv"

df = pd.read_csv(input_path)

X = df.iloc[:, :-1]
y = df.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y)

with mlflow.start_run():

    model = xgb.XGBClassifier()

    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    accuracy = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average="weighted")

    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("f1_score", f1)

    mlflow.sklearn.log_model(model, "model")

    print("Accuracy:", accuracy)

model_dir = os.environ["SM_MODEL_DIR"]

joblib.dump(model, f"{model_dir}/model.joblib")
