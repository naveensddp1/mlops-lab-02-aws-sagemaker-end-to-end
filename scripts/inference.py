import joblib
import os
import numpy as np


def model_fn(model_dir):
    """Load the saved XGBRegressor model."""
    model_path = os.path.join(model_dir, "model.joblib")
    model = joblib.load(model_path)
    print(f"Model loaded from {model_path}")
    return model


def input_fn(request_body, content_type):
    """Parse CSV input into a numpy array."""
    if content_type == "text/csv":
        # Handle single or multi-line CSV (no header)
        lines = request_body.strip().split("\n")
        parsed = []
        for line in lines:
            row = [float(x.strip()) for x in line.split(",")]
            parsed.append(row)
        return np.array(parsed)
    raise ValueError(f"Unsupported content type: {content_type}")


def predict_fn(input_data, model):
    """Run prediction."""
    return model.predict(input_data)


def output_fn(prediction, accept):
    """Return predictions as comma-separated string."""
    return ",".join(str(round(float(p), 4)) for p in prediction)