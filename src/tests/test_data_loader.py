from src.steps.data_loader import load_data

if __name__ == "__main__":
    df = load_data()
    print(df.head())
