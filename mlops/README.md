# Customer Churn Prediction

An end-to-end machine learning project for predicting whether a bank customer is likely to leave. The project preprocesses customer data, trains and evaluates classification models, tracks experiments with MLflow, registers the best model, and serves predictions through a FastAPI application.

## What the project does

- Prepares `data/Churn_Modeling.csv` by removing duplicate records and the `CustomerId` and `Surname` columns, then saves `data/processed_churn.csv`.
- Trains Logistic Regression, Random Forest, and Gradient Boosting classifiers.
- Logs accuracy, precision, recall, F1 score, model parameters, and model artifacts to a local MLflow tracking store.
- Registers the model with the highest F1 score as `churn_model` and assigns it the `champion` alias.
- Serves a churn prediction and churn probability from a FastAPI endpoint.

## Project structure

```text
mlops_demo/
├── app/
│   ├── main.py                    # FastAPI prediction service
│   └── utils.py                   # Request schema and feature preparation
├── data/
│   ├── Churn_Modeling.csv         # Raw customer churn dataset
│   └── processed_churn.csv        # Preprocessed dataset (generated)
├── mlruns/                        # MLflow artifacts
├── reports/                       # Report output directory
├── samples/                       # Existing sample payloads
├── src/
│   ├── data_preprocessing.py      # Churn dataset preprocessing
│   ├── model_tracking_mlflow.py   # Model training, evaluation, and registration
│   └── generate_report.py         # Legacy wine-data drift report script
├── mlflow.db                      # Local MLflow tracking database
└── requirements.txt               # Python dependencies
```

The files in `samples/` and `src/generate_report.py` are leftovers from the wine-quality demo: the sample payloads contain wine features, and the report script reads `processed_wine.csv`. They are not churn prediction examples or part of the churn pipeline.

## Setup

Run commands from the `mlops_demo` directory:

```bash
cd mlops_demo
python -m venv venv
```

Activate the virtual environment:

```powershell
# Windows PowerShell
.\venv\Scripts\Activate.ps1
```

```bash
# macOS/Linux
source venv/bin/activate
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

Make sure `data/Churn_Modeling.csv` is present before running the pipeline.

## Run the pipeline

### 1. Preprocess the data

```bash
python src/data_preprocessing.py
```

This creates `data/processed_churn.csv`.

### 2. Train and register models

```bash
python src/model_tracking_mlflow.py
```

The script logs three model runs to the `churn_experiment` experiment in `mlflow.db`, then registers the model with the best F1 score as `churn_model` and sets its `champion` alias.

### 3. View MLflow experiments

In another terminal, from `mlops_demo`, start the MLflow UI:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --host 127.0.0.1 --port 5001
```

Open [http://127.0.0.1:5001](http://127.0.0.1:5001) to inspect runs and metrics.

### 4. Start the prediction API

After training and registering a model, start the API from `mlops_demo`:

```bash
python app/main.py
```

The API loads `models:/churn_model@champion` by default. To load a different exported or registered model, set the `MODEL_URI` environment variable before starting the service.

Interactive API documentation is available at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).

## Make a prediction

Send a `POST` request to `/predict` with customer data. All fields are required:

```json
{
  "CreditScore": 619,
  "Geography": "France",
  "Gender": "Female",
  "Age": 42,
  "Tenure": 2,
  "Balance": 0.0,
  "NumOfProducts": 1,
  "HasCrCard": 1,
  "IsActiveMember": 1,
  "EstimatedSalary": 101348.88
}
```

The response contains a binary prediction (`1` means likely to churn; `0` means likely to stay) and the model's estimated probability of churn:

```json
{
  "prediction": 0,
  "churn_probability": 0.1234
}
```

The probability value above is illustrative; actual results depend on the trained model.
