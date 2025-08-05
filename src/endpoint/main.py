import pickle
from fastapi import FastAPI
from pydantic import BaseModel
#path = 'C:\\Users\\lupep\\PycharmProjects\\mlModelsFastAPIdp\\src\\model\\iris_model.pkl'
model_path = '..\\src\\model\\iris_model.pkl'
# Load the model once at startup
model = pickle.load(open(model_path, 'rb'))
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