from zenml import pipeline

from src.steps.data_loader import load_data
from src.steps.preprocess_split import preprocess_and_split
from src.steps.train_and_eval import train_and_evaluate_model
from src.steps.best_model_selector import compare_and_store_best
from src.steps.model_saver import save_model


@pipeline(enable_cache=False)
def anomaly_training_pipeline():
    """
    Training-only pipeline:
    - loads data
    - preprocesses
    - trains + evaluates model
    - compares with previous best
    - saves model only if better
    """

    # Step 1: Load data
    df = load_data()

    # Step 2: Preprocess + split
    X_train, X_test, y_test, preprocessor = preprocess_and_split(df)

    # Step 3: Train & evaluate
    model, metrics = train_and_evaluate_model(
        X_train=X_train,
        X_test=X_test,
        y_test=y_test,
    )

    # Step 4: Check if this model is better
    is_best = compare_and_store_best(metrics)

    # Step 5: Save model only if it's the best
    save_model(
        model=model,
        preprocessor=preprocessor,
        is_best=is_best,
    )
