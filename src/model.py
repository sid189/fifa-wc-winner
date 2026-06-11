"""Outcome model: Elo diff -> P(home win, draw, away win).

A multinomial logistic regression learns how Elo gaps translate into the
three-way outcome distribution observed in real matches. This gives us
draw probabilities (needed for the group stage) on top of raw Elo.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


CLASSES = np.array(["H", "D", "A"])


@dataclass
class OutcomeModel:
    clf: LogisticRegression
    class_order: np.ndarray  # order returned by clf.predict_proba

    def proba(self, elo_diff: float | np.ndarray) -> np.ndarray:
        x = np.atleast_2d(np.asarray(elo_diff, dtype=float)).T
        p = self.clf.predict_proba(x)
        # Reorder columns to (H, D, A) for ergonomics.
        idx = [list(self.class_order).index(c) for c in CLASSES]
        return p[:, idx]


def fit_outcome_model(snapshots: pd.DataFrame, min_date: str = "2006-01-01") -> OutcomeModel:
    df = snapshots[snapshots["date"] >= pd.Timestamp(min_date)].copy()
    X = df[["elo_diff"]].to_numpy()
    y = df["outcome"].to_numpy()
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X, y)
    return OutcomeModel(clf=clf, class_order=clf.classes_)


def match_proba(model: OutcomeModel, elo_a: float, elo_b: float, neutral: bool = True) -> dict:
    diff = (elo_a - elo_b) + (0.0 if neutral else 100.0)
    p = model.proba(np.array([diff]))[0]
    return {"H": float(p[0]), "D": float(p[1]), "A": float(p[2])}


def knockout_win_proba(model: OutcomeModel, elo_a: float, elo_b: float, neutral: bool = True) -> float:
    """P(A advances). Split draws 50/50 (penalty shootout proxy)."""
    p = match_proba(model, elo_a, elo_b, neutral=neutral)
    return p["H"] + 0.5 * p["D"]
