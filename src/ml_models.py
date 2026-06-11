"""Model zoo for the outcome (H/D/A) prediction task.

The current pipeline uses multinomial logistic regression on Elo diff. This
module lets you swap in gradient boosting, random forest, MLP, etc., and
compare honestly on held-out matches.

Feature set is intentionally small (Elo diff + a few easy adds) so
differences across algorithms reflect inductive bias rather than feature
engineering.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier

FEATURE_COLS = ["elo_diff", "abs_elo_diff", "is_friendly", "is_wc", "is_qualifier"]


def add_features(snapshots: pd.DataFrame) -> pd.DataFrame:
    """Append engineered columns to the snapshots DataFrame in place-safe way."""
    df = snapshots.copy()
    df["abs_elo_diff"] = df["elo_diff"].abs()
    t = df["tournament"].str.lower()
    df["is_friendly"] = t.str.contains("friendly").astype(int)
    df["is_wc"] = t.str.contains("fifa world cup").astype(int)
    df["is_qualifier"] = t.str.contains("qualification").astype(int)
    return df


def make_xy(snapshots: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    df = add_features(snapshots)
    X = df[FEATURE_COLS].to_numpy()
    y = df["outcome"].to_numpy()
    return X, y


def get_model_zoo(random_state: int = 42) -> dict:
    return {
        "logistic":        LogisticRegression(max_iter=1000),
        "gradient_boost":  GradientBoostingClassifier(
                              n_estimators=200, max_depth=3,
                              learning_rate=0.05, random_state=random_state),
        "random_forest":   RandomForestClassifier(
                              n_estimators=300, max_depth=8,
                              min_samples_leaf=20, random_state=random_state,
                              n_jobs=-1),
        "mlp":             MLPClassifier(
                              hidden_layer_sizes=(32, 16), max_iter=500,
                              random_state=random_state),
        "naive_bayes":     GaussianNB(),
    }
