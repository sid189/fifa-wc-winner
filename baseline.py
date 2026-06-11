"""End-to-end baseline: load results -> Elo -> outcome model -> 10k MC sims.

Run from the project root:
    pip install -r requirements.txt
    python baseline.py
"""
from __future__ import annotations

from src.data import load_results
from src.elo import compute_elo_history, top_n
from src.model import fit_outcome_model, match_proba
from src.simulate import monte_carlo
from src.groups_2026 import GROUPS_2026, HOST_TEAMS_2026, bracket_2026
from src.predictions import (
    group_stage_predictions,
    format_group_stage_table,
    draw_watch,
    most_likely_seeding,
    r32_predictions,
    format_r32_table,
    full_bracket_predictions,
    format_full_bracket_table,
)


def main() -> None:
    print("Loading match history...")
    results = load_results()
    print(f"  {len(results):,} matches "
          f"({results.date.min().date()} -> {results.date.max().date()})")

    print("Computing Elo ratings over full history...")
    ratings, snapshots = compute_elo_history(results, conmebol_qualifier_k_factor=0.5)
    print("\nCurrent top 15 Elo:")
    print(top_n(ratings, 15).to_string(index=False))

    print("\nFitting outcome model (Elo diff -> H/D/A)...")
    model = fit_outcome_model(snapshots, min_date="2006-01-01")

    print("\nSanity-check matchups (neutral ground):")
    for a, b in [("Argentina", "France"), ("Spain", "England"),
                 ("Brazil", "Germany")]:
        if a in ratings and b in ratings:
            p = match_proba(model, ratings[a], ratings[b], neutral=True)
            print(f"  {a} vs {b}: "
                  f"H={p['H']:.2f}  D={p['D']:.2f}  A={p['A']:.2f}")

    missing = [t for grp in GROUPS_2026.values() for t in grp if t not in ratings]
    if missing:
        print(f"\nWARNING: missing from Elo table -> {missing}")

    print("\nGroup-stage match-by-match forecasts:")
    gs = group_stage_predictions(GROUPS_2026, ratings, model,
                                 host_teams=HOST_TEAMS_2026)
    print(format_group_stage_table(gs))

    print("\n--- Matches most likely to end in a draw ---")
    print(draw_watch(gs, top_n=8).to_string(index=False))

    print("\nFull chalk-path bracket (favorite advances each round):")
    firsts, seconds, thirds = most_likely_seeding(GROUPS_2026, ratings,
                                                  host_teams=HOST_TEAMS_2026)
    bracket = bracket_2026(firsts, seconds, thirds)
    full = full_bracket_predictions(bracket, ratings, model)
    print(format_full_bracket_table(full))

    print("\nRunning 10,000 tournament simulations (real R32 bracket)...")
    predictions = monte_carlo(GROUPS_2026, ratings, model,
                              n_sims=10_000, seed=42,
                              host_teams=HOST_TEAMS_2026,
                              host_ko_boost=50.0,
                              bracket_fn=bracket_2026)

    print("\n2026 WORLD CUP - TOP 15 BY P(CHAMPION)")
    print(predictions.head(15).to_string(index=False))


if __name__ == "__main__":
    main()
