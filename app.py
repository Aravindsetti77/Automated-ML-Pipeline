from fastapi import FastAPI
import joblib
import numpy as np
from pydantic import BaseModel
model = joblib.load("model/iris_model.joblib")
app = FastAPI()
class iris_model(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float
@app.get("/")
def home():
    return {"message": "Iris Model Factory API is Online"}
@app.post("/Predict")
def predict(data:iris_model):
    features = np.array([[data.sepal_length, data.sepal_width, 
                          data.petal_length, data.petal_width]])
    prediction=model.predict(features)
    target_names = ['setosa', 'versicolor', 'virginica']
    result = target_names[int(prediction[0])]
    return {"prediction": result}