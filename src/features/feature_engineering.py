from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import pandas as pd


class FeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        # Log transforms
        for col in ["dur", "sbytes", "dbytes", "spkts", "dpkts"]:
            if col in X.columns:
                X[f"{col}_log"] = np.log1p(X[col])

        # Ratios
        if {"sbytes", "dbytes"}.issubset(X.columns):
            X["byte_ratio"] = X["sbytes"] / (X["dbytes"] + 1)

        if {"spkts", "dpkts"}.issubset(X.columns):
            X["pkt_ratio"] = X["spkts"] / (X["dpkts"] + 1)

        if {"sbytes", "dur"}.issubset(X.columns):
            X["byte_rate"] = X["sbytes"] / (X["dur"] + 1)

        return X
