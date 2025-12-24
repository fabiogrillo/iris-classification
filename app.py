from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

model = joblib.load("models/svm_v1.pkl")
scaler = joblib.load("models/std_scaler_v1.pkl")
class_names = ['setosa', 'versicolor', 'virginica']

app = FastAPI(title="Iris Classification API", version="1.0")

class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

class IrisOutput(BaseModel):
    prediction: int
    species: str

@app.get("/")
async def root():
    return {
        "message": "Welcome to the Iris Classification API"}

@app.post("/predict", response_model=IrisOutput)
async def predict(iris: IrisInput):
    input_data = np.array([[iris.sepal_length, iris.sepal_width, iris.petal_length, iris.petal_width]])
    input_data_scaled = scaler.transform(input_data)
    prediction = model.predict(input_data_scaled)
    species = class_names[prediction[0]]
    return IrisOutput(prediction=int(prediction[0]), species=species)