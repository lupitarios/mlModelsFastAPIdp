import os
import pickle
from fastapi import FastAPI
from pydantic import BaseModel

# Get the folder where the current script is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Go to the model folder relative to this file
MODEL_PATH = os.path.join(BASE_DIR, "..", "model", "iris_model.pkl")
# Normalize the path for Linux
MODEL_PATH = os.path.normpath(MODEL_PATH)

# Load the pickle model
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)
    print("Model loaded successfully Type")

app = FastAPI()

class IrisFeatures(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

# Endpoint prediction
@app.post('/predict')
def predict_iris(data: IrisFeatures):
    # FastAPI gives us an IrisFeatures instance with validated fields
    features = [[
        data.sepal_length,
        data.sepal_width,
        data.petal_length,
        data.petal_width
    ]]

    print("Features for prediction: ", features)
    print("Feature type: ", type(features))

    pred = model.predict(features)[0]
    print("Prediction type: ", type(pred))

    # Return the result as JSON
    return {'prediction': int(pred)}