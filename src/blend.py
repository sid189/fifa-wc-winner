"""Blend model predictions with bookmaker-implied probabilities.

The market aggregates lineup info, injuries, and tournament-mode performance
that pre-tournament Elo cannot see. A small market weight often improves
forecast accuracy at the cost of analytical "purity."

Final P = w * model + (1 - w) * market, renormalized so the blended
probabilities sum to 1 across the field.
"""
from __future__ import annotations

import pandas as pd


def blend_probabilities(
    model_preds: pd.DataFrame,
    market_probs: dict[str, float],
    w: float,
) -> pd.DataFrame:
    """Return a sorted DataFrame with model_p, market_p, blended_p columns.

    Teams without a market price get market_p = 0.0 (i.e., the model speaks
    alone for them); after renormalization their blended_p is effectively
    w * model_p.
    """
    if not 0.0 <= w <= 1.0:
        raise ValueError(f"w must be in [0, 1], got {w}")

    out = model_preds[["team", "p_champion"]].copy()
    out = out.rename(columns={"p_champion": "model_p"})
    out["market_p"] = out["team"].map(market_probs).fillna(0.0)
    raw = w * out["model_p"] + (1.0 - w) * out["market_p"]
    out["blended_p"] = raw / raw.sum()
    return out.sort_values("blended_p", ascending=False).reset_index(drop=True)


def blend_sweep(
    model_preds: pd.DataFrame,
    market_probs: dict[str, float],
    weights: list[float] | None = None,
) -> pd.DataFrame:
    """Compute blended probabilities for several values of w side-by-side.

    Returns a DataFrame with one column per weight: `w=0.00`, `w=0.25`, etc.
    Rows are teams. Useful for inspecting how much the prediction order
    changes as you weight the market vs the model.
    """
    weights = weights or [0.0, 0.25, 0.5, 0.75, 1.0]
    cols = {"team": model_preds["team"].to_list()}
    for w in weights:
        b = blend_probabilities(model_preds, market_probs, w)
        b = b.set_index("team")["blended_p"]
        cols[f"w={w:.2f}"] = [float(b.get(t, 0.0)) for t in cols["team"]]
    return pd.DataFrame(cols).sort_values(f"w={weights[len(weights)//2]:.2f}",
                                          ascending=False).reset_index(drop=True)
