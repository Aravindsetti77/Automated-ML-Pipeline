from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
import joblib
import pandas as pd
import yaml
import os

def load_config(config_path="config.yaml"):
    with open(config_path, "r") as file:
        return yaml.safe_load(file)

config = load_config()
app = FastAPI(title="Financial Prediction API")

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

@app.get("/")
def home():
    status = "Online" if model is not None else "Model Not Loaded"
    return {"message": "Financial Prediction API is Online.", "status": status}

@app.post("/Predict")
async def predict(request: Request):
    global model
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not trained yet. Run Script/train.py first.")
    
    data = await request.json()
    input_data = pd.DataFrame([data])
    
    try:
        prediction = model.predict(input_data)[0]
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction failed. Ensure inputs match the trained features. Error: {str(e)}")
        
    return {"prediction": round(float(prediction), 2)}

@app.post("/ReloadModel")
def reload_model():
    global model
    model_path = config['model']['model_path']
    if os.path.exists(model_path):
        model = joblib.load(model_path)
        return {"status": "success", "message": "Model hot-swapped successfully"}
    else:
        raise HTTPException(status_code=500, detail="Model file not found")