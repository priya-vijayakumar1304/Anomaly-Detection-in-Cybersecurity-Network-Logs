from src.steps.data_loader import load_data
from src.steps.preprocess_split import preprocess_and_split

if __name__ == "__main__":
    df = load_data()
    X_train, X_test, y_test, preprocessor = preprocess_and_split(df)

    print("X_train shape:", X_train.shape)
    print("X_test shape:", X_test.shape)
    print("y_test distribution:", {0: (y_test == 0).sum(), 1: (y_test == 1).sum()})
