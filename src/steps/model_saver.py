from zenml import step
import joblib
import os


@step
def save_model(
    model,
    preprocessor,
    is_best: bool,
    path: str = "artifacts/deployable_model.joblib",
) -> None:
    """
    Save model and preprocessor only if this run is the best so far.
    """

    if not is_best:
        print("[Model Saver] Model is not best. Nothing saved.")
        return

    os.makedirs(os.path.dirname(path), exist_ok=True)

    joblib.dump(
        {
            "model": model,
            "preprocessor": preprocessor,
        },
        path,
    )

    print(f"[Model Saver] Best model saved to {path}")
