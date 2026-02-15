import boto3
import os
import tarfile
import tempfile
import shutil
import sagemaker
from sagemaker.sklearn.model import SKLearnModel

region = os.environ["AWS_REGION"]
role = os.environ["SAGEMAKER_ROLE_ARN"]
endpoint_name = "wine-quality-endpoint"

print("Region:", region)
print("Role:", role)

boto_session = boto3.Session(region_name=region)
session = sagemaker.Session(boto_session=boto_session)
sm = boto_session.client("sagemaker")
s3 = boto_session.client("s3")

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
print("Original model artifact:", model_artifact)

# ── Repackage model.tar.gz with inference code ──────────────────
print("Repackaging model with inference code...")

bucket = model_artifact.split("/")[2]
key = "/".join(model_artifact.split("/")[3:])

tmpdir = tempfile.mkdtemp()
original_tar = os.path.join(tmpdir, "model_original.tar.gz")
extract_dir = os.path.join(tmpdir, "model_contents")
new_tar = os.path.join(tmpdir, "model.tar.gz")

# Download original model.tar.gz
s3.download_file(bucket, key, original_tar)
print("  Downloaded original model.tar.gz")

# Extract
os.makedirs(extract_dir, exist_ok=True)
with tarfile.open(original_tar, "r:gz") as tar:
    tar.extractall(extract_dir)
print("  Extracted contents:", os.listdir(extract_dir))

# Add code/ directory with inference script and minimal requirements
code_dir = os.path.join(extract_dir, "code")
if os.path.exists(code_dir):
    shutil.rmtree(code_dir)
os.makedirs(code_dir)

with open(os.path.join(code_dir, "inference.py"), "w") as f:
    f.write('''import joblib
import os
import numpy as np


def model_fn(model_dir):
    model_path = os.path.join(model_dir, "model.joblib")
    model = joblib.load(model_path)
    print(f"Model loaded from {model_path}")
    return model


def input_fn(request_body, content_type):
    if content_type == "text/csv":
        lines = request_body.strip().split("\\n")
        parsed = []
        for line in lines:
            row = [float(x.strip()) for x in line.split(",")]
            parsed.append(row)
        return np.array(parsed)
    raise ValueError(f"Unsupported content type: {content_type}")


def predict_fn(input_data, model):
    return model.predict(input_data)


def output_fn(prediction, accept):
    return ",".join(str(round(float(p), 4)) for p in prediction)
''')

with open(os.path.join(code_dir, "requirements.txt"), "w") as f:
    f.write("xgboost==2.0.3\n")

print("  Added code/inference.py and code/requirements.txt")

# Repackage
with tarfile.open(new_tar, "w:gz") as tar:
    for item in os.listdir(extract_dir):
        tar.add(os.path.join(extract_dir, item), arcname=item)
print("  Repackaged model.tar.gz")

# Upload
new_key = key.replace("output/model.tar.gz", "output/model_repackaged.tar.gz")
s3.upload_file(new_tar, bucket, new_key)
new_model_uri = f"s3://{bucket}/{new_key}"
print("  Uploaded to:", new_model_uri)

shutil.rmtree(tmpdir)

# ── Cleanup existing endpoint resources ─────────────────────────
print("Cleaning up any existing endpoint resources...")

try:
    sm.describe_endpoint(EndpointName=endpoint_name)
    # Endpoint exists — delete it
    sm.delete_endpoint(EndpointName=endpoint_name)
    print("  Deleting endpoint...")
    waiter = sm.get_waiter("endpoint_deleted")
    waiter.wait(EndpointName=endpoint_name, WaiterConfig={"Delay": 10, "MaxAttempts": 60})
    print("  Endpoint deletion confirmed")
except Exception:
    print("  No existing endpoint")

try:
    sm.delete_endpoint_config(EndpointConfigName=endpoint_name)
    print("  Deleted endpoint config")
except Exception:
    print("  No existing endpoint config")

try:
    models = sm.list_models(SortBy="CreationTime", SortOrder="Descending", MaxResults=10)
    for m in models["Models"]:
        if "wine" in m["ModelName"].lower() or "sagemaker-scikit" in m["ModelName"].lower():
            sm.delete_model(ModelName=m["ModelName"])
            print(f"  Deleted old model: {m['ModelName']}")
except Exception:
    pass

# ── Deploy — NO source_dir needed, code is inside model.tar.gz ──
import time
print("Waiting 30s for cleanup to propagate...")
time.sleep(30)
print("Deploying new endpoint...")

model = SKLearnModel(
    model_data=new_model_uri,
    role=role,
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