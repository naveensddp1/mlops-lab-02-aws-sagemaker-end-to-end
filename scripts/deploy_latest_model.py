import boto3
import os
import time
import sagemaker

from sagemaker.model import Model
from sagemaker.image_uris import retrieve

# ENV variables from GitHub Actions
region = os.environ["AWS_REGION"]
role = os.environ["SAGEMAKER_ROLE_ARN"]

endpoint_name = "wine-quality-endpoint"

print("Region:", region)
print("Role:", role)

# Create SageMaker session
session = sagemaker.Session()

# Create boto3 client
sm = boto3.client("sagemaker", region_name=region)

print("Finding latest completed training job...")

# Get latest completed training job
jobs = sm.list_training_jobs(
    SortBy="CreationTime",
    SortOrder="Descending",
    MaxResults=20
)["TrainingJobSummaries"]

latest_job = None

for job in jobs:
    if job["TrainingJobStatus"] == "Completed":
        latest_job = job["TrainingJobName"]
        break

if latest_job is None:
    raise Exception("No completed training job found")

print("Using training job:", latest_job)

# Get model artifact location
job_details = sm.describe_training_job(
    TrainingJobName=latest_job
)

model_artifact = job_details["ModelArtifacts"]["S3ModelArtifacts"]

print("Model artifact:", model_artifact)

# Get correct sklearn container image
image_uri = retrieve(
    framework="sklearn",
    region=region,
    version="1.2-1",
    py_version="py3",
    instance_type="ml.m5.large"
)

print("Using image:", image_uri)

# Create SageMaker Model object
model = Model(
    model_data=model_artifact,
    role=role,
    entry_point="inference.py",
    source_dir="scripts",
    image_uri=image_uri,
    sagemaker_session=session
)

print("Deleting existing endpoint if exists...")

# Delete old endpoint safely
try:
    session.delete_endpoint(endpoint_name)
    print("Old endpoint deleted")
except:
    print("No existing endpoint")

print("Deploying new endpoint...")

# Deploy endpoint
predictor = model.deploy(
    endpoint_name=endpoint_name,
    instance_type="ml.m5.large",
    initial_instance_count=1,
    wait=True
)

print("SUCCESS — endpoint deployed:", endpoint_name)
