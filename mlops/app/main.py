import mlflow
import pandas as pd
from fastapi import FastAPI
import uvicorn
import os

# # Import helpers from utils
# from utils import CustomerData, engineer_features, align_features

# # Point to the SQLite database
# mlflow.set_tracking_uri("sqlite:///mlflow.db")

# # Load model interactive selection at startup
# # model, selected_run_id = select_model("churn_experiment")
# model = mlflow.sklearn.load_model("models:/churn_model/champion")
# app = FastAPI(title=f"Churn Prediction API (Model: {model.__class__.__name__})")

# @app.get("/")
# def root():
#     return {"status": "active", "loaded_run_id": None}

# @app.post("/predict")
# def predict_churn(customer_data: CustomerData):
#     df = pd.DataFrame([customer_data.model_dump()])

#     # Step 2 & 3: Feature engineering and schema alignment
#     df = engineer_features(df)
#     df = align_features(df, model)

#     prediction = model.predict(df)[0]
#     churn_probability = model.predict_proba(df)[0][1]  # Probability of class '1' (churn)
    
#     # Return directly as string since target is string-binned ("low", "medium", "high")
#     return {"prediction": str(prediction), "churn_probability": churn_probability}

# if __name__ == "__main__":
#     uvicorn.run(app, host="127.0.0.1", port=8000)

from utils import CustomerData, engineer_features, align_features
# Point to the SQLite database
mlflow.set_tracking_uri("sqlite:///mlflow.db")
# Load the registered champion model at startup (the Docker image sets MODEL_URI to an exported copy)
MODEL_URI = os.getenv("MODEL_URI", "models:/churn_model@champion")
model = mlflow.sklearn.load_model(MODEL_URI)
app = FastAPI(title="Customer Churn Prediction API")
@app.get("/")
def root():
    return {"status": "active", "model_uri": MODEL_URI}
@app.post("/predict")
def predict(customer: CustomerData):
    df = pd.DataFrame([customer.model_dump()])
    # Encode raw fields and align columns with the model's training schema
    df = engineer_features(df)
    df = align_features(df, model)
    prediction = int(model.predict(df)[0])
    churn_probability = float(model.predict_proba(df)[0][1])
    # prediction: 1 = likely to churn, 0 = likely to stay
    return {"prediction": prediction, "churn_probability": round(churn_probability, 4)}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)