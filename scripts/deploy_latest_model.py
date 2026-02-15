# scripts/deploy_latest_model.py

import boto3
import os
import time

import sagemaker
from sagemaker.sklearn.model import SKLearnModel

region = os.environ["AWS_REGION"]
role = os.environ["SAGEMAKER_ROLE_ARN"]

endpoint_name = "wine-quality-endpoint"

sm = boto3.client("sagemaker", region_name=region)

print("Finding latest training job...")

jobs = sm.list_training_jobs(
    SortBy="CreationTime",
    SortOrder="Descending",
    MaxResults=10
)["TrainingJobSummaries"]

latest_job = None

for job in jobs:
    if job["TrainingJobStatus"] == "Completed":
        latest_job = job["TrainingJobName"]
        break

if not latest_job:
    raise Exception("No completed training job found")

print("Using training job:", latest_job)

job_details = sm.describe_training_job(
    TrainingJobName=latest_job
)

model_artifact = job_details["ModelArtifacts"]["S3ModelArtifacts"]

print("Model artifact:", model_artifact)

# SageMaker session
session = sagemaker.Session()

# THIS IS THE CRITICAL FIX
model = SKLearnModel(
    model_data=model_artifact,
    role=role,
    entry_point="scripts/inference.py",   # VERY IMPORTANT
    framework_version="1.2-1",
    py_version="py3",
    sagemaker_session=session
)

print("Deleting old endpoint if exists...")

try:
    model.sagemaker_session.delete_endpoint(endpoint_name)
except:
    pass

print("Deploying endpoint...")

predictor = model.deploy(
    endpoint_name=endpoint_name,
    instance_type="ml.m5.large",
    initial_instance_count=1
)

print("SUCCESS — endpoint deployed:", endpoint_name)
