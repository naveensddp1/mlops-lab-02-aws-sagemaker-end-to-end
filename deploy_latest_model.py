import boto3
import os
import time
import sagemaker
from sagemaker.sklearn.model import SKLearnModel

# ENV variables from GitHub Actions
region = os.environ["AWS_REGION"]
role = os.environ["SAGEMAKER_ROLE_ARN"]

endpoint_name = "wine-quality-endpoint"

print("Region:", region)
print("Role:", role)

# Create SageMaker session
boto_session = boto3.Session(region_name=region)
session = sagemaker.Session(boto_session=boto_session)
sm = boto_session.client("sagemaker")

# ── Find latest completed training job ──────────────────────────
print("Finding latest completed training job...")

jobs = sm.list_training_jobs(
    SortBy="CreationTime",
    SortOrder="Descending",
    MaxResults=20,
)["TrainingJobSummaries"]

latest_job = None
for job in jobs:
    if job["TrainingJobStatus"] == "Completed":
        latest_job = job["TrainingJobName"]
        break

if latest_job is None:
    raise Exception("No completed training job found")

print("Using training job:", latest_job)

job_details = sm.describe_training_job(TrainingJobName=latest_job)
model_artifact = job_details["ModelArtifacts"]["S3ModelArtifacts"]
print("Model artifact:", model_artifact)

# ── Delete existing endpoint + config + model (if any) ──────────
print("Cleaning up any existing endpoint resources...")

# 1) Delete endpoint
try:
    sm.delete_endpoint(EndpointName=endpoint_name)
    print("  Deleted endpoint")
    # Wait until endpoint is fully gone
    waiter = sm.get_waiter("endpoint_deleted")
    waiter.wait(
        EndpointName=endpoint_name,
        WaiterConfig={"Delay": 10, "MaxAttempts": 60},
    )
    print("  Endpoint deletion confirmed")
except sm.exceptions.ClientError:
    print("  No existing endpoint")

# 2) Delete endpoint config
try:
    sm.delete_endpoint_config(EndpointConfigName=endpoint_name)
    print("  Deleted endpoint config")
except sm.exceptions.ClientError:
    print("  No existing endpoint config")

# 3) Delete old model (best-effort, name may differ)
try:
    sm.delete_model(ModelName=endpoint_name)
    print("  Deleted old model")
except sm.exceptions.ClientError:
    print("  No existing model to delete")

# ── Create and deploy new model ─────────────────────────────────
print("Deploying new endpoint...")

model = SKLearnModel(
    model_data=model_artifact,
    role=role,
    entry_point="inference.py",
    source_dir="scripts",
    framework_version="1.2-1",
    py_version="py3",
    sagemaker_session=session,
)

predictor = model.deploy(
    endpoint_name=endpoint_name,
    instance_type="ml.m5.large",
    initial_instance_count=1,
    wait=True,
)

print("SUCCESS — endpoint deployed:", endpoint_name)