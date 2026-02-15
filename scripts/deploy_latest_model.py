# scripts/deploy_latest_model.py

import boto3
import os
import time

region = os.environ["AWS_REGION"]
role = os.environ["SAGEMAKER_ROLE_ARN"]

sm = boto3.client("sagemaker", region_name=region)

endpoint_name = "wine-quality-endpoint"

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

model_name = f"wine-model-{int(time.time())}"

container = {
    "Image": "720646828776.dkr.ecr.ap-south-1.amazonaws.com/sagemaker-scikit-learn:1.2-1-cpu-py3",
    "ModelDataUrl": model_artifact,
}

print("Creating model...")

sm.create_model(
    ModelName=model_name,
    ExecutionRoleArn=role,
    PrimaryContainer=container
)

endpoint_config = f"{model_name}-config"

print("Creating endpoint config...")

sm.create_endpoint_config(
    EndpointConfigName=endpoint_config,
    ProductionVariants=[
        {
            "VariantName": "AllTraffic",
            "ModelName": model_name,
            "InstanceType": "ml.m5.large",
            "InitialInstanceCount": 1
        }
    ]
)

print("Deploying endpoint...")

try:
    sm.delete_endpoint(EndpointName=endpoint_name)
    time.sleep(10)
except:
    pass

sm.create_endpoint(
    EndpointName=endpoint_name,
    EndpointConfigName=endpoint_config
)

print("Waiting for endpoint...")

while True:

    status = sm.describe_endpoint(
        EndpointName=endpoint_name
    )["EndpointStatus"]

    print("Status:", status)

    if status == "InService":
        break

    if status == "Failed":
        raise Exception("Endpoint deployment failed")

    time.sleep(30)

print("SUCCESS — endpoint ready:", endpoint_name)
