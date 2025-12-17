from zenml import step
import mlflow
import mlflow.sklearn
from typing import Dict, Tuple
from typing_extensions import Annotated

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, roc_auc_score


@step(experiment_tracker="anomaly_mlflow_tracker")
def train_and_evaluate_model(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> Tuple[
    Annotated[IsolationForest, "IsolationForest_Model"],
    Annotated[Dict[str, float], "Classification_report"],
]:

    """
    Train IsolationForest on NORMAL data only and evaluate on test set.

    Returns:
    - trained IsolationForest model
    - evaluation metrics dict
    """

    # Model definition
    model = IsolationForest(
        n_estimators=300,
        max_samples=1.0,
        contamination=0.3,
        random_state=42,
        n_jobs=-1,
    )

    # Train (unsupervised)
    model.fit(X_train)

    # Evaluate on test set
    scores = model.decision_function(X_test)
    raw_preds = model.predict(X_test)

    # Convert IF output: -1 → anomaly(1), 1 → normal(0)
    y_pred = [1 if p == -1 else 0 for p in raw_preds]

    roc_auc = roc_auc_score(y_test, -scores)
    report = classification_report(y_test, y_pred, output_dict=True)

    metrics = {
        "roc_auc": roc_auc,
        "precision_normal": report["0"]["precision"],
        "recall_normal": report["0"]["recall"],
        "f1_normal": report["0"]["f1-score"],
        "precision_anomaly": report["1"]["precision"],
        "recall_anomaly": report["1"]["recall"],
        "f1_anomaly": report["1"]["f1-score"],
    }

    # Log parameters, metrics and model to MLflow
    mlflow.log_params({
        "n_estimators": 300,
        "max_samples": 1.0,
        "contamination": 0.3,
        "model_type": "IsolationForest",
    })

    for k, v in metrics.items():
        mlflow.log_metric(k, v)
    
    mlflow.sklearn.log_model(
    sk_model=model,
    artifact_path="model",
    )

    print("Model training & evaluation completed")
    print("ROC-AUC:", roc_auc)

    return model, metrics
