from src.steps.data_loader import load_data
from src.steps.preprocess_split import preprocess_and_split
from src.steps.train_and_eval import train_and_evaluate_model

if __name__ == "__main__":
    df = load_data()
    X_train, X_test, y_test, preprocessor = preprocess_and_split(df)
    model, metrics = train_and_evaluate_model(X_train, X_test, y_test)

    print("Returned metrics:")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
