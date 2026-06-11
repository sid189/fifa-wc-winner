"""Test whether halving K for CONMEBOL qualifiers fixes Brazil-2002.

Background: in `backtest_all.py`, Brazil 2002 was predicted rank #9 with P=2.8%
despite a stacked squad (Ronaldo/Ronaldinho/Rivaldo). One hypothesis is that
the CONMEBOL qualification format - a brutal 18-game round-robin - deflates
big-team Elo when they stumble against weak opposition. Halving K for those
matches should partially undo that deflation.

This script runs the multi-year backtest twice (default K vs halved CONMEBOL K)
and reports champion rank per year for each variant.

Run:
    python experiment_conmebol_k.py
"""
from __future__ import annotations

import pandas as pd

from src.data import load_results
from src.elo import compute_elo_history
from src.model import fit_outcome_model
from src.simulate import monte_carlo
from backtest_all import CONFIGS


def run_variant(all_results: pd.DataFrame, k_factor: float) -> list[dict]:
    out = []
    for cfg in CONFIGS:
        pre = all_results[all_results["date"] < pd.Timestamp(cfg.start_date)
                          ].reset_index(drop=True)
        ratings, snapshots = compute_elo_history(
            pre, conmebol_qualifier_k_factor=k_factor)
        train_start = pd.Timestamp(cfg.start_date) - pd.DateOffset(years=16)
        model = fit_outcome_model(snapshots, min_date=str(train_start.date()))
        preds = monte_carlo(
            cfg.groups, ratings, model,
            n_sims=10_000, seed=42,
            n_third_qualifiers=0,
            bracket_fn=cfg.bracket_fn,
        )
        champ = cfg.actual["Champion"]
        rank_idx = preds.index[preds["team"] == champ]
        rank = int(rank_idx[0]) + 1 if len(rank_idx) else None
        pc = float(preds.loc[rank_idx[0], "p_champion"]) if len(rank_idx) else 0.0
        out.append({
            "year": cfg.year,
            "champion": champ,
            "rank": rank,
            "p_champion": pc,
        })
    return out


def main() -> None:
    print("Loading match history (slow)...")
    all_results = load_results()

    print("\n=== Variant A: default K (no override) ===")
    a = run_variant(all_results, k_factor=1.0)

    print("\n=== Variant B: K halved for CONMEBOL qualifiers ===")
    b = run_variant(all_results, k_factor=0.5)

    print("\n" + "=" * 70)
    print("CONMEBOL K-FACTOR COMPARISON")
    print("=" * 70)
    rows = []
    for ra, rb in zip(a, b):
        rows.append({
            "year": ra["year"],
            "champion": ra["champion"],
            "rank_default": ra["rank"],
            "p_default": f"{ra['p_champion']:.3f}",
            "rank_halved": rb["rank"],
            "p_halved": f"{rb['p_champion']:.3f}",
            "rank_delta": (ra["rank"] - rb["rank"]) if ra["rank"] and rb["rank"] else None,
        })
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))

    hits_a = sum(1 for r in a if r["rank"] and r["rank"] <= 6)
    hits_b = sum(1 for r in b if r["rank"] and r["rank"] <= 6)
    print(f"\nChampion-in-top-6 hit rate: default {hits_a}/{len(a)}  "
          f"vs halved {hits_b}/{len(b)}")
    if hits_b > hits_a:
        print("VERDICT: halving K for CONMEBOL qualifiers improves the backtest.")
        print("  Apply the override to the 2026 prediction by passing")
        print("  `conmebol_qualifier_k_factor=0.5` to compute_elo_history in baseline.py.")
    elif hits_b == hits_a:
        print("VERDICT: same hit count. Check rank_delta to see if individual")
        print("  rankings improved without changing the top-6 threshold.")
    else:
        print("VERDICT: halving K makes things worse. The 2002 Brazil failure")
        print("  is not due to CONMEBOL Elo deflation; look elsewhere "
              "(squad features, tournament-mode model).")


if __name__ == "__main__":
    main()
