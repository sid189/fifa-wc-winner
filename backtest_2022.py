"""Honest backtest: freeze the pipeline at 2022-11-19 and predict the 2022 WC.

Actual 2022 results:
    Champion : Argentina
    Runner-up: France
    Semi-fin : Croatia, Morocco

A useful baseline puts Argentina/France/Brazil meaningfully high
(P(champion) > 5% each) and ranks the eventual champion top ~6.
Surprises like Morocco's SF run are not expected to be predicted.

Run:
    python backtest_2022.py
"""
from __future__ import annotations

import pandas as pd

from src.data import load_results
from src.elo import compute_elo_history, top_n
from src.model import fit_outcome_model
from src.simulate import monte_carlo
from src.groups_2022 import GROUPS_2022, bracket_2022

WC2022_START = "2022-11-20"

ACTUAL = {
    "Champion": "Argentina",
    "Runner-up": "France",
    "Semi-finalists": ["Croatia", "Morocco"],
    "Quarter-finalists": ["Netherlands", "Brazil", "England", "Portugal"],
}


def main() -> None:
    print("Loading match history...")
    results = load_results()
    pre = results[results["date"] < pd.Timestamp(WC2022_START)].reset_index(drop=True)
    print(f"  Using {len(pre):,} matches up to {WC2022_START} "
          f"(dropped {len(results) - len(pre):,} from 2022 WC onwards)")

    print("\nComputing Elo as of 2022-11-19...")
    ratings, snapshots = compute_elo_history(pre)
    print(top_n(ratings, 15).to_string(index=False))

    print("\nFitting outcome model on 2006 -> 2022-11-19 matches...")
    model = fit_outcome_model(snapshots, min_date="2006-01-01")

    missing = [t for grp in GROUPS_2022.values() for t in grp if t not in ratings]
    if missing:
        print(f"WARNING: missing Elo for: {missing}")

    print("\nRunning 10,000 simulations of the 2022 tournament...")
    preds = monte_carlo(
        GROUPS_2022, ratings, model,
        n_sims=10_000, seed=42,
        n_third_qualifiers=0,
        bracket_fn=bracket_2022,
    )

    print("\n=== PREDICTED P(CHAMPION) - TOP 12 ===")
    print(preds.head(12).to_string(index=False))

    actual_champ = ACTUAL["Champion"]
    rank = preds.index[preds["team"] == actual_champ]
    if len(rank):
        rk = int(rank[0]) + 1
        pc = float(preds.loc[rank[0], "p_champion"])
        ps = float(preds.loc[rank[0], "p_semi"])
        print(f"\nActual champion {actual_champ}: ranked #{rk} by model "
              f"(P_champ={pc:.3f}, P_semi={ps:.3f})")
    else:
        print(f"\n{actual_champ} missing from predictions table - check name match.")

    print("\nActual 2022 result for reference:")
    for k, v in ACTUAL.items():
        print(f"  {k}: {v}")

    print("\nVerdict heuristics:")
    print("  - Strong signal if Argentina ranks top 6 in P(champion)")
    print("  - Strong signal if Argentina + France + Brazil each have P(champion) > 5%")
    print("  - Surprises (Morocco SF) are not expected to be predicted")


if __name__ == "__main__":
    main()
