"""Test whether squad-value Elo augmentation fixes Italy-2006 and Brazil-2002.

Runs the multi-year backtest twice:
  Variant A: Elo only (alpha=0)        - baseline
  Variant B: Elo + 30 * log(squad_val) - augmented

If Variant B moves Italy 2006 from #8 toward the top 5 and Brazil 2002 from #9
toward the top 5 *without* breaking 2010/2014/2018, the augmentation is a real
fix and should be enabled in baseline.py.

Data is limited to what is in data/squad_values.csv. Years without coverage
fall back to no augmentation (and behave like Variant A).

Run:
    python experiment_squad_value.py
"""
from __future__ import annotations

import pandas as pd

from src.data import load_results
from src.elo import compute_elo_history
from src.model import fit_outcome_model
from src.simulate import monte_carlo
from src.squad_values import augment_elo, load_squad_values
from backtest_all import CONFIGS


def run_variant(all_results: pd.DataFrame, alpha: float) -> list[dict]:
    sv = load_squad_values()
    out = []
    for cfg in CONFIGS:
        pre = all_results[all_results["date"] < pd.Timestamp(cfg.start_date)
                          ].reset_index(drop=True)
        ratings, snapshots = compute_elo_history(pre)
        train_start = pd.Timestamp(cfg.start_date) - pd.DateOffset(years=16)
        model = fit_outcome_model(snapshots, min_date=str(train_start.date()))

        if alpha > 0:
            field = [t for grp in cfg.groups.values() for t in grp]
            ratings = augment_elo(ratings, cfg.year, teams=field, alpha=alpha, df=sv)

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
        out.append({"year": cfg.year, "champion": champ, "rank": rank, "p_champion": pc})
    return out


def main() -> None:
    print("Loading match history...")
    all_results = load_results()

    print("\n=== Variant A: Elo only ===")
    a = run_variant(all_results, alpha=0.0)

    print("\n=== Variant B: Elo + 30 * log(squad_value) ===")
    b = run_variant(all_results, alpha=30.0)

    rows = []
    for ra, rb in zip(a, b):
        delta = (ra["rank"] - rb["rank"]) if ra["rank"] and rb["rank"] else None
        rows.append({
            "year": ra["year"],
            "champion": ra["champion"],
            "rank_elo": ra["rank"],
            "p_elo": f"{ra['p_champion']:.3f}",
            "rank_aug": rb["rank"],
            "p_aug": f"{rb['p_champion']:.3f}",
            "rank_delta": delta,
        })

    print("\n" + "=" * 70)
    print("SQUAD-VALUE AUGMENTATION COMPARISON")
    print("=" * 70)
    print(pd.DataFrame(rows).to_string(index=False))

    hits_a = sum(1 for r in a if r["rank"] and r["rank"] <= 6)
    hits_b = sum(1 for r in b if r["rank"] and r["rank"] <= 6)
    print(f"\nChampion-in-top-6 hit rate: Elo-only {hits_a}/{len(a)}  "
          f"vs augmented {hits_b}/{len(b)}")

    if hits_b > hits_a:
        print("VERDICT: squad-value augmentation helps. Apply by importing")
        print("  `augment_elo` in baseline.py and calling it on `ratings` before")
        print("  passing into monte_carlo().")
    elif hits_b == hits_a:
        print("VERDICT: same hit count. Check rank_delta to see if the failed")
        print("  years (Brazil-2002, Italy-2006) moved at all.")
    else:
        print("VERDICT: augmentation makes things worse. Either the data is too")
        print("  sparse (most years lack coverage in data/squad_values.csv) or")
        print("  squad value is not the missing signal.")


if __name__ == "__main__":
    main()
