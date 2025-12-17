from zenml import step
import pandas as pd
from typing_extensions import Annotated


@step
def load_data() -> Annotated[pd.DataFrame, "Clean_df"]:
    """
    - Loads UNSW_NB15 dataset
    - Performs only basic cleaning
    """

    df = pd.read_csv("data/UNSW_NB15_training-set.csv")

    df = df.drop(columns=["attack_cat", "id"], errors="ignore")

    # Fix categorical column
    if "service" in df.columns:
        df["service"] = df["service"].replace("-", "unknown")

    if "label" not in df.columns:
        raise ValueError("Target column 'label' not found")

    print("Data loaded & basic cleaning completed")
    print("Shape:", df.shape)

    return df
