"""Compare model predictions against bookmaker outright odds.

The market aggregates millions of bets and is genuinely hard to beat - if
your model assigns 22% champion probability to a team the market prices at
10%, the burden is on you to explain why.

Workflow:
1. Collect current decimal outright odds from a low-margin book (Pinnacle
   or Betfair Exchange are best; Bet365/William Hill add a wider vig).
2. Pass them in as a {team: decimal_odds} dict.
3. `implied_from_decimal` devigs to a proper probability distribution.
4. `compare_to_market` puts model vs market side-by-side with diff and ratio.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def implied_from_decimal(odds: dict[str, float]) -> dict[str, float]:
    """Decimal odds -> devigged implied probabilities (sum to 1.0).

    Equal-weighting devig (proportional). Cleanest when the book is low-margin;
    for high-margin books consider Shin's method instead.
    """
    raw = {team: 1.0 / o for team, o in odds.items()}
    total = sum(raw.values())
    return {team: p / total for team, p in raw.items()}


def compare_to_market(
    predictions: pd.DataFrame,
    market: dict[str, float],
    top_n: int = 20,
) -> pd.DataFrame:
    """Side-by-side model vs market for teams in `market` dict.

    Adds diff (model - market) and ratio (model / market). Positive diff means
    the model thinks the team is undervalued by the market (your potential
    bet); negative means overvalued.
    """
    rows = []
    for team, market_p in market.items():
        sel = predictions.loc[predictions["team"] == team, "p_champion"]
        model_p = float(sel.iloc[0]) if len(sel) else 0.0
        rows.append({
            "team": team,
            "model_p": model_p,
            "market_p": market_p,
            "diff": model_p - market_p,
            "ratio": (model_p / market_p) if market_p > 0 else np.nan,
        })
    return (
        pd.DataFrame(rows)
        .sort_values("diff", ascending=False)
        .reset_index(drop=True)
        .head(top_n)
    )


def plot_calibration(comparison: pd.DataFrame, ax=None):
    """Scatter: market_p on x, model_p on y, with diagonal reference line."""
    import matplotlib.pyplot as plt
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(comparison["market_p"], comparison["model_p"], s=60, alpha=0.7)
    for _, r in comparison.iterrows():
        ax.annotate(r["team"], (r["market_p"], r["model_p"]),
                    fontsize=8, xytext=(4, 4), textcoords="offset points")
    lim = max(comparison["market_p"].max(), comparison["model_p"].max()) * 1.1
    ax.plot([0, lim], [0, lim], "k--", alpha=0.4, label="Model = Market")
    ax.set_xlabel("Market implied P(champion)")
    ax.set_ylabel("Model P(champion)")
    ax.set_title("Model vs market calibration\n(above line = model overrates, below = underrates)")
    ax.legend()
    ax.grid(alpha=0.3)
    return ax
