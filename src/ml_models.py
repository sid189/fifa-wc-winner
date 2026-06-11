"""Model zoo for the outcome (H/D/A) prediction task.

The current pipeline uses multinomial logistic regression on Elo diff. This
module lets you swap in gradient boosting, random forest, MLP, XGBoost,
LightGBM, CatBoost, etc., and compare honestly on held-out matches.

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
from sklearn.preprocessing import LabelEncoder

# Optional boosting libraries. Skip silently if not installed.
try:
    from xgboost import XGBClassifier  # type: ignore
    _HAS_XGB = True
except ImportError:
    _HAS_XGB = False

try:
    from lightgbm import LGBMClassifier  # type: ignore
    _HAS_LGB = True
except ImportError:
    _HAS_LGB = False

try:
    from catboost import CatBoostClassifier  # type: ignore
    _HAS_CAT = True
except ImportError:
    _HAS_CAT = False


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


class StringLabelWrapper:
    """Adapter that lets a classifier with numeric-only labels accept strings.

    XGBoost, LightGBM, and CatBoost expect numeric y. This wrapper handles the
    encoding internally so the rest of the pipeline can keep using "H"/"D"/"A".
    Exposes `.classes_`, `.predict`, `.predict_proba`, and `.fit` matching the
    sklearn convention.
    """

    def __init__(self, base):
        self.base = base
        self.encoder = LabelEncoder()

    def fit(self, X, y):
        y_enc = self.encoder.fit_transform(y)
        self.base.fit(X, y_enc)
        return self

    @property
    def classes_(self):
        return self.encoder.classes_

    def predict(self, X):
        return self.encoder.inverse_transform(self.base.predict(X))

    def predict_proba(self, X):
        return self.base.predict_proba(X)


def get_model_zoo(random_state: int = 42) -> dict:
    """Return the available models. Boosting libs skipped if not installed."""
    zoo: dict = {
        "logistic":       LogisticRegression(max_iter=1000),
        "gradient_boost": GradientBoostingClassifier(
                             n_estimators=200, max_depth=3,
                             learning_rate=0.05, random_state=random_state),
        "random_forest":  RandomForestClassifier(
                             n_estimators=300, max_depth=8,
                             min_samples_leaf=20, random_state=random_state,
                             n_jobs=-1),
        "mlp":            MLPClassifier(
                             hidden_layer_sizes=(32, 16), max_iter=500,
                             random_state=random_state),
        "naive_bayes":    GaussianNB(),
    }

    if _HAS_XGB:
        zoo["xgboost"] = StringLabelWrapper(
            XGBClassifier(
                n_estimators=300, max_depth=4, learning_rate=0.05,
                objective="multi:softprob", eval_metric="mlogloss",
                random_state=random_state, n_jobs=-1, verbosity=0,
            )
        )

    if _HAS_LGB:
        zoo["lightgbm"] = StringLabelWrapper(
            LGBMClassifier(
                n_estimators=300, max_depth=-1, num_leaves=15,
                learning_rate=0.05, random_state=random_state,
                n_jobs=-1, verbosity=-1,
            )
        )

    if _HAS_CAT:
        zoo["catboost"] = StringLabelWrapper(
            CatBoostClassifier(
                iterations=300, depth=4, learning_rate=0.05,
                random_seed=random_state, verbose=0, allow_writing_files=False,
            )
        )

    return zoo


def available_models() -> list[str]:
    """List of model names that will be trained in `get_model_zoo`."""
    return list(get_model_zoo().keys())
