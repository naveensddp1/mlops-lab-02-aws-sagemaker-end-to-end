# scripts/create_pipeline.py

import boto3
import sagemaker
import os
from sagemaker.estimator import Estimator
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import TrainingStep

region = os.environ["AWS_REGION"]
role = os.environ["SAGEMAKER_ROLE_ARN"]
bucket = os.environ["S3_BUCKET"]
mlflow_uri = os.environ["MLFLOW_TRACKING_URI"]

session = sagemaker.Session()

estimator = Estimator(
    image_uri=sagemaker.image_uris.retrieve(
        "xgboost",
        region,
        version="1.7-1"
    ),
    role=role,
    instance_count=1,
    instance_type="ml.m5.large",
    output_path=f"s3://{bucket}/models/",
    environment={
        "MLFLOW_TRACKING_URI": mlflow_uri
    }
)

estimator.set_hyperparameters()

train_input = sagemaker.inputs.TrainingInput(
    f"s3://{bucket}/data/",
    content_type="text/csv"
)

step_train = TrainingStep(
    name="TrainWineModel",
    estimator=estimator,
    inputs={"train": train_input}
)

pipeline = Pipeline(
    name="wine-mlflow-pipeline",
    steps=[step_train],
    sagemaker_session=session
)

pipeline.upsert(role_arn=role)

execution = pipeline.start()

print("Pipeline started:", execution.arn)
