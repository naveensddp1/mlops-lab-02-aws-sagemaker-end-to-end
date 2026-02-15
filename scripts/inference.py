import joblib
import os
import numpy as np


def model_fn(model_dir):

    model = joblib.load(os.path.join(model_dir, "model.joblib"))
    return model


def input_fn(request_body, content_type):

    if content_type == "text/csv":

        data = np.array([float(x) for x in request_body.split(",")])
        return data.reshape(1, -1)

    raise Exception("Unsupported content type")


def predict_fn(input_data, model):

    prediction = model.predict(input_data)

    return prediction


def output_fn(prediction, content_type):

    return str(prediction[0])
