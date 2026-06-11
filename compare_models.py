"""Train 5 classifiers on pre-2022 matches, evaluate on 2022 pre-WC matches.

Same task (H/D/A multinomial), same features, same train/test split. The
question: how much of the Argentina-#2 win in the 2022 backtest is from
the Elo signal vs the choice of logistic regression?

Metrics:
- log_loss : proper scoring rule, lower is better
- brier    : multi-class Brier, lower is better
- accuracy : top-1 accuracy (less meaningful with ~25% draw base rate)

Run:
    python compare_models.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss

from src.data import load_results
from src.elo import compute_elo_history
from src.ml_models import get_model_zoo, make_xy, FEATURE_COLS

CLASSES = ["H", "D", "A"]


def multiclass_brier(y_true: np.ndarray, p_pred: np.ndarray) -> float:
    """Multi-class Brier score (Gneiting & Raftery). Lower is better."""
    onehot = np.zeros_like(p_pred)
    for i, c in enumerate(CLASSES):
        onehot[:, i] = (y_true == c).astype(int)
    return float(((p_pred - onehot) ** 2).sum(axis=1).mean())


def main() -> None:
    print("Loading match history...")
    results = load_results()
    pre = results[results["date"] < pd.Timestamp("2022-11-20")].reset_index(drop=True)

    print("Computing Elo...")
    _, snapshots = compute_elo_history(pre)
    snapshots = snapshots[snapshots["date"] >= pd.Timestamp("2006-01-01")]

    train = snapshots[snapshots["date"] < pd.Timestamp("2021-06-01")]
    test = snapshots[snapshots["date"] >= pd.Timestamp("2021-06-01")]
    print(f"  Train: {len(train):,} matches  ({train.date.min().date()} -> {train.date.max().date()})")
    print(f"  Test:  {len(test):,} matches  ({test.date.min().date()} -> {test.date.max().date()})")
    print(f"  Features: {FEATURE_COLS}")

    X_train, y_train = make_xy(train)
    X_test, y_test = make_xy(test)

    rows = []
    for name, model in get_model_zoo().items():
        print(f"  fitting {name}...")
        model.fit(X_train, y_train)
        p = model.predict_proba(X_test)
        class_order = list(model.classes_)
        col_idx = [class_order.index(c) for c in CLASSES]
        p_ordered = p[:, col_idx]
        ll = log_loss(y_test, p_ordered, labels=CLASSES)
        br = multiclass_brier(y_test, p_ordered)
        acc = accuracy_score(y_test, model.predict(X_test))
        rows.append({"model": name, "log_loss": ll, "brier": br, "accuracy": acc})

    df = pd.DataFrame(rows).sort_values("log_loss")
    print("\n=== MODEL COMPARISON (log_loss = lower better) ===")
    print(df.to_string(index=False))

    print("\nReading the table:")
    print("  - logistic is the current baseline; if anything beats it materially, swap it in.")
    print("  - if gradient_boost wins by >0.01 log_loss, features have non-linear signal.")
    print("  - if all models tie within 0.005, you're feature-limited, not model-limited;")
    print("    add features (form, market value, xG) before switching algorithms.")


if __name__ == "__main__":
    main()
