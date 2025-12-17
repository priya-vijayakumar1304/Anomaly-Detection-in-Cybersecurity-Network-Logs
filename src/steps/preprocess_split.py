from zenml import step
from typing import Tuple
from typing_extensions import Annotated

import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.features.feature_engineering import FeatureEngineer


@step
def preprocess_and_split(
    df: pd.DataFrame,
) -> Tuple[
    Annotated[np.ndarray, "X_train"],
    Annotated[np.ndarray, "X_test"],
    Annotated[np.ndarray, "y_test"],
    Annotated[Pipeline, "preprocessor"],
]:

    if "label" not in df.columns:
        raise ValueError("Target column 'label' not found")

    y = df["label"].values
    X = df.drop(columns=["label"])

    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
    numeric_cols = X.select_dtypes(exclude=["object"]).columns.tolist()

    print(f"Categorical columns: {len(categorical_cols)}")
    print(f"Numeric columns: {len(numeric_cols)}")

    column_transformer = ColumnTransformer(
        transformers=[
            ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), categorical_cols),
            ("num", RobustScaler(), numeric_cols),
        ],
        remainder="drop",
    )

    # FULL preprocessing pipeline
    preprocessor = Pipeline(
        steps=[
            ("feature_engineering", FeatureEngineer()),
            ("column_transformer", column_transformer),
        ]
    )

    X_train_df, X_test_df, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42,
    )

    X_train_normal = X_train_df[y_train == 0]

    print(f"Training samples (normal only): {X_train_normal.shape[0]}")
    print(f"Test samples: {X_test_df.shape[0]}")

    X_train = preprocessor.fit_transform(X_train_normal)
    X_test = preprocessor.transform(X_test_df)

    return X_train, X_test, y_test, preprocessor
