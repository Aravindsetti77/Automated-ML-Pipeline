from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import joblib
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
import yaml
import os

def load_config(config_path="config.yaml"):
    with open(config_path, "r") as file:
        return yaml.safe_load(file)

config = load_config()
app = FastAPI(title="Explainable Credit Card Fraud Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model_path = config['model']['model_path']
if os.path.exists(model_path):
    model = joblib.load(model_path)
else:
    model = None
    print(f"Warning: Model not found at {model_path}. API will return 503 until trained.")

class Transaction(BaseModel):
    distance_from_home: float = Field(ge=0.0)
    distance_from_last_transaction: float = Field(ge=0.0)
    ratio_to_median_purchase_price: float = Field(ge=0.0)
    repeat_retailer: int = Field(ge=0, le=1)
    used_chip: int = Field(ge=0, le=1)
    used_pin_number: int = Field(ge=0, le=1)
    online_order: int = Field(ge=0, le=1)

class ContributedTransaction(Transaction):
    fraud: int = Field(ge=0, le=1)

@app.get("/")
def home():
    status = "Online" if model is not None else "Model Not Loaded"
    return {"message": "Credit Card Fraud Detection API is Online.", "status": status}

@app.post("/Predict")
def predict(data: Transaction):
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not trained yet. Run Script/train.py first.")
    
    input_data = pd.DataFrame([data.model_dump()])
    
    prediction = model.predict(input_data)[0]
    
    try:
        probability = model.predict_proba(input_data)[0][1] * 100
    except:
        probability = 99.9 if prediction == 1 else 0.1
        
    label = "Fraud" if prediction == 1 else "Legitimate"
    return {"prediction": label, "class": int(prediction), "probability": round(probability, 2)}

@app.post("/ContributeData")
def contribute_data(data: ContributedTransaction):
    live_data_path = "Data/live_data.csv"
    df = pd.DataFrame([data.model_dump()])
    
    if os.path.exists(live_data_path):
        df.to_csv(live_data_path, mode='a', header=False, index=False)
    else:
        df.to_csv(live_data_path, mode='w', header=True, index=False)
        
    return {"status": "success", "message": "Data recorded securely"}

@app.post("/ReloadModel")
def reload_model():
    global model
    model_path = config['model']['model_path']
    if os.path.exists(model_path):
        model = joblib.load(model_path)
        return {"status": "success", "message": "Model hot-swapped successfully"}
    else:
        raise HTTPException(status_code=500, detail="Model file not found")