from zenml import step
from typing import Dict
import json
import os
from typing_extensions import Annotated

BEST_METRICS_PATH = "best_metrics.json"


@step
def compare_and_store_best(metrics: Dict[str, float]) -> Annotated[bool, "Is_Best"]:
    """
    Compare current model metrics with the previous best.
    Returns True if current model is better and should be saved.
    """

    current_roc = metrics.get("roc_auc", 0.0)

    # Load previous best metrics if they exist
    if os.path.exists(BEST_METRICS_PATH):
        with open(BEST_METRICS_PATH, "r") as f:
            best_metrics = json.load(f)
        best_roc = best_metrics.get("roc_auc", 0.0)
    else:
        best_roc = 0.0

    print(f"[Model Comparison] Current ROC-AUC: {current_roc:.4f}")
    print(f"[Model Comparison] Best ROC-AUC so far: {best_roc:.4f}")

    if current_roc > best_roc:
        print("[Model Comparison] New best model found. Updating best_metrics.json")
        with open(BEST_METRICS_PATH, "w") as f:
            json.dump(metrics, f, indent=2)
        return True

    print("[Model Comparison] Current model is NOT better. Skipping save.")
    return False
