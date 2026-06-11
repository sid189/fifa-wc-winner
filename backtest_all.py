"""Multi-tournament backtest: 2014, 2018, 2022.

Three independent freeze-and-predict runs. Single-tournament backtest
(Argentina #2 in 2022) could be luck; checking 3 in a row tells us whether
the methodology is reliable or fitted to one Cup.

Verdict heuristic: model is sound if the eventual champion is in the
predicted top 6 in *all three* tournaments.

Run:
    python backtest_all.py
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable
import pandas as pd

from src.data import load_results
from src.elo import compute_elo_history
from src.model import fit_outcome_model
from src.simulate import monte_carlo
from src.groups_2014 import GROUPS_2014, bracket_2014, ACTUAL_2014
from src.groups_2018 import GROUPS_2018, bracket_2018, ACTUAL_2018
from src.groups_2022 import GROUPS_2022, bracket_2022

ACTUAL_2022 = {
    "Champion": "Argentina",
    "Runner-up": "France",
    "Semi-finalists": ["Croatia", "Morocco"],
}


@dataclass
class TournamentConfig:
    year: int
    start_date: str
    groups: dict[str, list[str]]
    bracket_fn: Callable
    actual: dict


CONFIGS = [
    TournamentConfig(2014, "2014-06-12", GROUPS_2014, bracket_2014, ACTUAL_2014),
    TournamentConfig(2018, "2018-06-14", GROUPS_2018, bracket_2018, ACTUAL_2018),
    TournamentConfig(2022, "2022-11-20", GROUPS_2022, bracket_2022, ACTUAL_2022),
]


def run_backtest(cfg: TournamentConfig, all_results: pd.DataFrame, n_sims: int = 10_000) -> dict:
    pre = all_results[all_results["date"] < pd.Timestamp(cfg.start_date)].reset_index(drop=True)
    ratings, snapshots = compute_elo_history(pre)
    model = fit_outcome_model(snapshots, min_date="2006-01-01")

    missing = [t for grp in cfg.groups.values() for t in grp if t not in ratings]
    if missing:
        print(f"  WARNING ({cfg.year}): missing Elo for {missing}")

    preds = monte_carlo(
        cfg.groups, ratings, model,
        n_sims=n_sims, seed=42,
        n_third_qualifiers=0,
        bracket_fn=cfg.bracket_fn,
    )

    champ = cfg.actual["Champion"]
    rank_idx = preds.index[preds["team"] == champ]
    rank = int(rank_idx[0]) + 1 if len(rank_idx) else None
    pc = float(preds.loc[rank_idx[0], "p_champion"]) if len(rank_idx) else 0.0
    ps = float(preds.loc[rank_idx[0], "p_semi"]) if len(rank_idx) else 0.0

    return {
        "year": cfg.year,
        "actual_champion": champ,
        "predicted_rank": rank,
        "predicted_p_champion": pc,
        "predicted_p_semi": ps,
        "top5": preds.head(5)[["team", "p_champion"]].values.tolist(),
    }


def main() -> None:
    print("Loading match history...")
    all_results = load_results()

    summary = []
    for cfg in CONFIGS:
        print(f"\n=== BACKTEST {cfg.year} (freeze at {cfg.start_date}) ===")
        out = run_backtest(cfg, all_results)
        print(f"  Predicted top-5: {out['top5']}")
        print(f"  Actual champion {out['actual_champion']}: rank #{out['predicted_rank']} "
              f"(P_champ={out['predicted_p_champion']:.3f})")
        summary.append(out)

    print("\n" + "=" * 60)
    print("MULTI-TOURNAMENT SUMMARY")
    print("=" * 60)
    df = pd.DataFrame(summary)[["year", "actual_champion", "predicted_rank",
                                "predicted_p_champion", "predicted_p_semi"]]
    print(df.to_string(index=False))

    pass_count = sum(1 for r in summary if r["predicted_rank"] and r["predicted_rank"] <= 6)
    print(f"\nChampion-in-top-6 hit rate: {pass_count}/{len(summary)}")
    if pass_count == len(summary):
        print("VERDICT: methodology is reliable across tournaments.")
    elif pass_count >= 2:
        print("VERDICT: signal is real but noisy. Investigate failures before trusting 2026.")
    else:
        print("VERDICT: 2022 was likely lucky. Model needs work before 2026.")


if __name__ == "__main__":
    main()
