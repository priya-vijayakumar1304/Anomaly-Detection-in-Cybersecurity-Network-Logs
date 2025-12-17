from typing import Tuple
from zenml import step
import joblib
import os
from typing_extensions import Annotated

@step
def load_deployable_model(
    path: str = "artifacts/deployable_model.joblib",
) -> Tuple[
    Annotated[object, "model"],
    Annotated[object, "preprocessor"]
]:
    
    """
    Load the best model + preprocessor saved during training.
    """

    if not os.path.exists(path):
        raise RuntimeError(
            f"No deployable model found at {path}. "
            "Run training pipeline first."
        )

    artifact = joblib.load(path)

    model = artifact["model"]
    preprocessor = artifact["preprocessor"]

    print(f"[Deployment Loader] Loaded model from {path}")

    return model, preprocessor
