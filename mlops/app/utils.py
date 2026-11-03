import sys
import mlflow
import mlflow.sklearn
import pandas as pd

from fastapi import FastAPI
from pydantic import BaseModel

from typing import Literal

# # class CustomerData(BaseModel):
#   CreditScore: int
#   Geography: Literal["France", "Germany", "Spain"]
#   Gender: Literal["Female", "Male"]
#   Age: int
#   Tenure: int
#   Balance: float
#   NumOfProducts: int
#   HasCrCard: int
#   IsActiveMember: int
#   EstimatedSalary: float

# def select_model(experiment_name: str = "wine_experiment"):
#     """Lists MLflow runs for the given experiment and loads the selected model."""
#     client = mlflow.tracking.MlflowClient()
    
#     experiment = client.get_experiment_by_name(experiment_name)
#     if not experiment:
#         print(f"Experiment '{experiment_name}' not found in mlflow.db.")
#         sys.exit(1)
        
#     runs = client.search_runs(experiment_ids=[experiment.experiment_id])
#     if not runs:
#         print("No logged model runs found in experiment.")
#         sys.exit(1)
        
#     print(f"\n--- Available Models in MLFlow ({experiment_name}) ---")
#     for idx, run in enumerate(runs):
#         run_id = run.info.run_id
#         model_type = run.data.params.get("model_type", "Unknown")
#         accuracy = run.data.metrics.get("accuracy", "N/A")
#         print(f"[{idx}] Run ID: {run_id} | Type: {model_type} | Accuracy: {accuracy}")
        
#     choice = int(input("\nSelect model index to load into app: "))
#     selected_run = runs[choice]
#     run_id = selected_run.info.run_id
    
#     print(f"\nLoading model from Run ID: {run_id}...")
#     model_uri = f"runs:/{run_id}/model"
#     loaded_model = mlflow.sklearn.load_model(model_uri)
#     return loaded_model, run_id


# def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
#     """Encode raw customer fields exactly like src/data_preprocessing.py does for training."""
#     df["Gender"] = df["Gender"].map({"Female": 0, "Male": 1})
#     df["Geography_Germany"] = (df["Geography"] == "Germany").astype(int)   
#     df["Geography_Spain"] = (df["Geography"] == "Spain").astype(int)
#     return df.drop(columns=["Geography"])
   
# def align_features(df: pd.DataFrame, loaded_model) -> pd.DataFrame:
#     """Ensure exact feature presence and order matching the model schema."""
#     if hasattr(loaded_model, "feature_names_in_"):
#         expected_cols = list(loaded_model.feature_names_in_)
#         for col in expected_cols:
#             if col not in df.columns:
#                 df[col] = 0
#         df = df[expected_cols]
#     return df


class CustomerData(BaseModel):
    """One bank customer: the raw Churn_Modeling.csv fields, without identifiers and the target."""
    CreditScore: int
    Geography: Literal["France", "Germany", "Spain"]
    Gender: Literal["Female", "Male"]
    Age: int
    Tenure: int
    Balance: float
    NumOfProducts: int
    HasCrCard: int
    IsActiveMember: int
    EstimatedSalary: float

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Encode raw customer fields exactly like src/data_preprocessing.py does for training."""
    df["Gender"] = df["Gender"].map({"Female": 0, "Male": 1})
    df["Geography_Germany"] = (df["Geography"] == "Germany").astype(int)
    df["Geography_Spain"] = (df["Geography"] == "Spain").astype(int)
    return df.drop(columns=["Geography"])

def align_features(df: pd.DataFrame, loaded_model) -> pd.DataFrame:
    """Ensure exact feature presence and order matching the model schema."""
    if hasattr(loaded_model, "feature_names_in_"):
        expected_cols = list(loaded_model.feature_names_in_)
        for col in expected_cols:
            if col not in df.columns:
                df[col] = 0
        df = df[expected_cols]
    return df   