from zenml import step
import pandas as pd
import numpy as np
from typing_extensions import Annotated

@step
def load_data() -> Annotated[pd.DataFrame,"Clean_df"]:
    """
    - Loads UNSW_NB15 dataset
    - Perform data cleaning
    - Feature engineering
    """

    # Load dataset
    df = pd.read_csv("data/UNSW_NB15_training-set.csv")

    # Make a clean copy
    df_prep = df.copy()

    # Drop leakage / unnecessary columns
    df_prep = df_prep.drop(columns=["attack_cat", "id"], errors="ignore")

    #fixing categorical column
    df_prep['service'] = df_prep['service'].replace('-', 'unknown', regex=True)

    # Ensure target exists
    if "label" not in df_prep.columns:
        raise ValueError("Target column 'label' not found")

    # Log transformation for skewed features
    log_features = ["dur", "sbytes", "dbytes", "spkts", "dpkts"]
    for col in log_features:
        if col in df_prep.columns:
            df_prep[col + "_log"] = np.log1p(df_prep[col])

    # Feature engineering
    if all(c in df_prep.columns for c in ["sbytes", "dbytes"]):
        df_prep["byte_ratio"] = df_prep["sbytes"] / (df_prep["dbytes"] + 1)

    if all(c in df_prep.columns for c in ["spkts", "dpkts"]):
        df_prep["pkt_ratio"] = df_prep["spkts"] / (df_prep["dpkts"] + 1)

    if all(c in df_prep.columns for c in ["sbytes", "dur"]):
        df_prep["byte_rate"] = df_prep["sbytes"] / (df_prep["dur"] + 1)

    print("Data loaded & feature engineering completed")
    print("Shape:", df_prep.shape)

    return df_prep
