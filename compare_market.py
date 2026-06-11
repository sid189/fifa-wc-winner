"""Model vs bookmaker outright odds for the 2026 WC.

REPLACE `CURRENT_ODDS` with fresh decimal odds before running. Pinnacle is
recommended (low margin). Source suggestions:
- https://www.pinnacle.com/en/soccer/world-cup/matchups
- https://www.betfair.com/exchange/plus/football/competition/28741761
- https://www.oddsportal.com/soccer/world/world-cup-2026/outrights/

Higher diff (model - market) means the model thinks the team is undervalued.
Treat large gaps as either (a) genuine alpha or (b) a model bug. The current
top-15 output shows Spain 22% vs typical market ~12%; that gap is the entry
point for the comparative study.
"""
from __future__ import annotations

import pandas as pd

from src.data import load_results
from src.elo import compute_elo_history
from src.model import fit_outcome_model
from src.simulate import monte_carlo
from src.groups_2026 import GROUPS_2026, HOST_TEAMS_2026, bracket_2026
from src.market import implied_from_decimal, compare_to_market


# Decimal odds (template - REPLACE with current values from a real book).
# These are rough placeholders to illustrate the workflow.
CURRENT_ODDS: dict[str, float] = {
    "Spain":         8.0,
    "Argentina":     8.5,
    "France":        9.0,
    "England":      10.0,
    "Brazil":       11.0,
    "Germany":      15.0,
    "Portugal":     17.0,
    "Netherlands":  19.0,
    "Belgium":      26.0,
    "Italy":        29.0,
    "Croatia":      41.0,
    "Uruguay":      51.0,
    "Colombia":     51.0,
    "Switzerland":  81.0,
    "United States": 81.0,
    "Mexico":      151.0,
    "Japan":       151.0,
    "Morocco":     151.0,
    "Ecuador":     251.0,
    "Norway":      301.0,
}


def main() -> None:
    print("Building 2026 prediction (this takes ~30s)...")
    results = load_results()
    ratings, snapshots = compute_elo_history(results)
    model = fit_outcome_model(snapshots, min_date="2006-01-01")
    preds = monte_carlo(
        GROUPS_2026, ratings, model,
        n_sims=10_000, seed=42,
        host_teams=HOST_TEAMS_2026,
        bracket_fn=bracket_2026,
    )

    print("\nDevigging market odds...")
    market_probs = implied_from_decimal(CURRENT_ODDS)
    overround = sum(1.0 / o for o in CURRENT_ODDS.values())
    print(f"  Book overround: {overround:.3f} "
          f"(1.0 = no vig; >1.05 typical for sportsbooks)")

    print("\n=== MODEL vs MARKET ===")
    cmp = compare_to_market(preds, market_probs, top_n=20)
    fmt = cmp.copy()
    fmt["model_p"] = fmt["model_p"].map(lambda x: f"{x:.1%}")
    fmt["market_p"] = fmt["market_p"].map(lambda x: f"{x:.1%}")
    fmt["diff"] = fmt["diff"].map(lambda x: f"{x:+.1%}")
    fmt["ratio"] = fmt["ratio"].map(lambda x: f"{x:.2f}x")
    print(fmt.to_string(index=False))

    print("\nInterpretation:")
    print("  positive diff -> model rates team HIGHER than market (potential value bet)")
    print("  negative diff -> model rates team LOWER (market sees something we don't)")
    print("\nTo also save a scatter plot:")
    print("  from src.market import plot_calibration")
    print("  import matplotlib.pyplot as plt")
    print("  plot_calibration(cmp); plt.savefig('figures/market_calibration.png')")


if __name__ == "__main__":
    main()
