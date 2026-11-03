from pathlib import Path
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Use SQLite backend in project root
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("churn_experiment")

REGISTERED_MODEL_NAME = "churn_model"

DATA_DIR = Path('data')
df = pd.read_csv(DATA_DIR / 'processed_churn.csv')

df["Gender"] = LabelEncoder().fit_transform(df["Gender"])
df = pd.get_dummies(df , columns = ["Geography"] , drop_first = True)

x = df.drop("Exited" , axis = 1)
y = df["Exited"]
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define models to train and evaluate
models = {
    'Logistic Regression': LogisticRegression(),
    'Random Forest': RandomForestClassifier(random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42)
}

# Define trusted types for skops tree serialization
SKOPS_TRUSTED_TYPES = ["sklearn.tree._tree.Tree"]

# Train, evaluate, and log each model in its own MLflow run
results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    # acc = model.score(X_test, y_test)
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred)
    }

    with mlflow.start_run(run_name=name) as run:
        mlflow.log_param("model_type", name)
        mlflow.log_metrics(metrics) 

        if hasattr(model, "get_params"):
            mlflow.log_params(model.get_params())

        # Pass skops_trusted_types and use name instead of deprecated artifact_path
        model_info =mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            skops_trusted_types=SKOPS_TRUSTED_TYPES
        )
        
        results[name] = {"f1": metrics["f1"], "model_uri": model_info.model_uri}
       # print(f"[{name}] Logged run {run.info.run_id} with Accuracy: {acc:.4f}")
        print(f"[{name}] Logged run {run.info.run_id} | " + " | ".join([f"{k}: {v:.4f}" for k, v in metrics.items()]))

# Register the best model by F1 
best_name = max(results, key=lambda n: results[n]["f1"])
version = mlflow.register_model(results[best_name]["model_uri"], REGISTERED_MODEL_NAME)
# Point the 'champion' alias at the new version so serving can load models:/churn_model@champion
mlflow.MlflowClient().set_registered_model_alias(REGISTERED_MODEL_NAME, "champion", version.version)
print(f"\nRegistered {best_name} as '{REGISTERED_MODEL_NAME}' version {version.version} (alias: champion)")