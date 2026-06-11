"""End-to-end: compute model preds, devig market odds, blend, report.

Run:
    python blend_market.py

Output:
    - top-10 ranking at each blend weight (w = 0, 0.25, 0.5, 0.75, 1)
    - per-team table: model_p, market_p, blended_p for each w
"""
from __future__ import annotations

import pandas as pd

from src.data import load_results
from src.elo import compute_elo_history
from src.model import fit_outcome_model
from src.simulate import monte_carlo
from src.groups_2026 import GROUPS_2026, HOST_TEAMS_2026, bracket_2026
from src.market import implied_from_decimal
from src.blend import blend_sweep, blend_probabilities

# Re-use the same odds source as report_market.py.
from report_market import CURRENT_ODDS


def main() -> None:
    print("Running 2026 prediction pipeline (slow)...")
    results = load_results()
    ratings, snapshots = compute_elo_history(results)
    model = fit_outcome_model(snapshots, min_date="2006-01-01")
    preds = monte_carlo(
        GROUPS_2026, ratings, model,
        n_sims=10_000, seed=42,
        host_teams=HOST_TEAMS_2026,
        host_ko_boost=50.0,
        bracket_fn=bracket_2026,
    )

    market_probs = implied_from_decimal(CURRENT_ODDS)

    print("\n=== BLEND SWEEP - top 10 across weights ===")
    sweep = blend_sweep(preds, market_probs,
                        weights=[0.0, 0.25, 0.5, 0.75, 1.0])
    pct_cols = [c for c in sweep.columns if c.startswith("w=")]
    fmt = sweep.copy()
    for c in pct_cols:
        fmt[c] = fmt[c].map(lambda x: f"{x:.1%}")
    print(fmt.head(10).to_string(index=False))

    print("\nReading the columns:")
    print("  w=0.00 -> 100% market, 0% model (raw bookmaker view)")
    print("  w=0.50 -> equal blend (recommended starting point)")
    print("  w=1.00 -> 100% model, 0% market (our raw output)")

    print("\n=== EQUAL-WEIGHT BLEND - full table top 15 ===")
    blended = blend_probabilities(preds, market_probs, w=0.5)
    fmt2 = blended.head(15).copy()
    for c in ["model_p", "market_p", "blended_p"]:
        fmt2[c] = fmt2[c].map(lambda x: f"{x:.1%}")
    print(fmt2.to_string(index=False))


if __name__ == "__main__":
    main()
