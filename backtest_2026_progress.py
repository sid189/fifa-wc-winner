"""Final backtest: score the model against the completed 2026 WC.

Unlike backtest_2022.py (written after the fact against historical data
already in the results file), this scores the tournament using the exact
ratings/model report.py produced on 2026-06-11 -- frozen pre-tournament,
because load_results() drops the NA-score 2026 WC rows that live in
data/results.csv (data.py:35).

Actual results below were sourced from FIFA's match centre, ESPN and
Al Jazeera live coverage (cross-checked across all three) on 2026-07-20,
the day after the tournament concluded. Full 104-match tournament:
Spain beat Argentina 1-0 (AET, Ferran Torres 106') in the final at
MetLife Stadium; England beat France 6-4 in the third-place playoff.

Run:
    python backtest_2026_progress.py
"""
from __future__ import annotations

import pandas as pd

from src.data import load_results
from src.elo import compute_elo_history, top_n
from src.model import fit_outcome_model, match_proba, knockout_win_proba
from src.simulate import HOST_BOOST
from src.groups_2026 import HOST_TEAMS_2026

# ---------------------------------------------------------------------------
# Actual results, as played (source: en.wikipedia.org raw wikitext, group
# pages + round-of-32 + knockout-stage pages, retrieved 2026-07-10).
# score_a/score_b are the final score (after extra time if played).
# shootout_winner is set only when the match was drawn after ET and decided
# on penalties.
# ---------------------------------------------------------------------------

GROUP_STAGE = [
    # Group A
    ("Mexico", "South Africa", 2, 0), ("South Korea", "Czech Republic", 2, 1),
    ("Czech Republic", "South Africa", 1, 1), ("Mexico", "South Korea", 1, 0),
    ("Czech Republic", "Mexico", 0, 3), ("South Africa", "South Korea", 1, 0),
    # Group B
    ("Canada", "Bosnia and Herzegovina", 1, 1), ("Qatar", "Switzerland", 1, 1),
    ("Switzerland", "Bosnia and Herzegovina", 4, 1), ("Canada", "Qatar", 6, 0),
    ("Switzerland", "Canada", 2, 1), ("Bosnia and Herzegovina", "Qatar", 3, 1),
    # Group C
    ("Brazil", "Morocco", 1, 1), ("Haiti", "Scotland", 0, 1),
    ("Scotland", "Morocco", 0, 1), ("Brazil", "Haiti", 3, 0),
    ("Scotland", "Brazil", 0, 3), ("Morocco", "Haiti", 4, 2),
    # Group D
    ("United States", "Paraguay", 4, 1), ("Australia", "Turkey", 2, 0),
    ("United States", "Australia", 2, 0), ("Turkey", "Paraguay", 0, 1),
    ("Turkey", "United States", 3, 2), ("Paraguay", "Australia", 0, 0),
    # Group E
    ("Germany", "Curaçao", 7, 1), ("Ivory Coast", "Ecuador", 1, 0),
    ("Germany", "Ivory Coast", 2, 1), ("Ecuador", "Curaçao", 0, 0),
    ("Curaçao", "Ivory Coast", 0, 2), ("Ecuador", "Germany", 2, 1),
    # Group F
    ("Netherlands", "Japan", 2, 2), ("Sweden", "Tunisia", 5, 1),
    ("Netherlands", "Sweden", 5, 1), ("Tunisia", "Japan", 0, 4),
    ("Japan", "Sweden", 1, 1), ("Tunisia", "Netherlands", 1, 3),
    # Group G
    ("Belgium", "Egypt", 1, 1), ("Iran", "New Zealand", 2, 2),
    ("Belgium", "Iran", 0, 0), ("New Zealand", "Egypt", 1, 3),
    ("Egypt", "Iran", 1, 1), ("New Zealand", "Belgium", 1, 5),
    # Group H
    ("Spain", "Cape Verde", 0, 0), ("Saudi Arabia", "Uruguay", 1, 1),
    ("Spain", "Saudi Arabia", 4, 0), ("Uruguay", "Cape Verde", 2, 2),
    ("Cape Verde", "Saudi Arabia", 0, 0), ("Uruguay", "Spain", 0, 1),
    # Group I
    ("France", "Senegal", 3, 1), ("Iraq", "Norway", 1, 4),
    ("France", "Iraq", 3, 0), ("Norway", "Senegal", 3, 2),
    ("Norway", "France", 1, 4), ("Senegal", "Iraq", 5, 0),
    # Group J
    ("Argentina", "Algeria", 3, 0), ("Austria", "Jordan", 3, 1),
    ("Argentina", "Austria", 2, 0), ("Jordan", "Algeria", 1, 2),
    ("Algeria", "Austria", 3, 3), ("Jordan", "Argentina", 1, 3),
    # Group K
    ("Portugal", "DR Congo", 1, 1), ("Uzbekistan", "Colombia", 1, 3),
    ("Portugal", "Uzbekistan", 5, 0), ("Colombia", "DR Congo", 1, 0),
    ("Colombia", "Portugal", 0, 0), ("DR Congo", "Uzbekistan", 3, 1),
    # Group L
    ("England", "Croatia", 4, 2), ("Ghana", "Panama", 1, 0),
    ("England", "Ghana", 0, 0), ("Panama", "Croatia", 0, 1),
    ("Panama", "England", 0, 2), ("Croatia", "Ghana", 2, 1),
]

# (team_a, team_b, score_a, score_b, shootout_winner_or_None)
# score is final after extra time where applicable.
KNOCKOUTS = [
    # Round of 32
    ("r32", "South Africa", "Canada", 0, 1, None),
    ("r32", "Brazil", "Japan", 2, 1, None),
    ("r32", "Germany", "Paraguay", 1, 1, "Paraguay"),
    ("r32", "Netherlands", "Morocco", 1, 1, "Morocco"),
    ("r32", "Ivory Coast", "Norway", 1, 2, None),
    ("r32", "France", "Sweden", 3, 0, None),
    ("r32", "Mexico", "Ecuador", 2, 0, None),
    ("r32", "England", "DR Congo", 2, 1, None),
    ("r32", "Belgium", "Senegal", 3, 2, None),
    ("r32", "United States", "Bosnia and Herzegovina", 2, 0, None),
    ("r32", "Spain", "Austria", 3, 0, None),
    ("r32", "Portugal", "Croatia", 2, 1, None),
    ("r32", "Switzerland", "Algeria", 2, 0, None),
    ("r32", "Australia", "Egypt", 1, 1, "Egypt"),
    ("r32", "Argentina", "Cape Verde", 3, 2, None),
    ("r32", "Colombia", "Ghana", 1, 0, None),
    # Round of 16
    ("r16", "Canada", "Morocco", 0, 3, None),
    ("r16", "Paraguay", "France", 0, 1, None),
    ("r16", "Brazil", "Norway", 1, 2, None),
    ("r16", "Mexico", "England", 2, 3, None),
    ("r16", "Portugal", "Spain", 0, 1, None),
    ("r16", "United States", "Belgium", 1, 4, None),
    ("r16", "Argentina", "Egypt", 3, 2, None),
    ("r16", "Switzerland", "Colombia", 0, 0, "Switzerland"),
    # Quarterfinals
    ("qf", "France", "Morocco", 2, 0, None),
    ("qf", "Spain", "Belgium", 2, 1, None),
    ("qf", "England", "Norway", 2, 1, None),
    ("qf", "Argentina", "Switzerland", 3, 1, None),
    # Semifinals
    ("sf", "Spain", "France", 2, 0, None),
    ("sf", "Argentina", "England", 2, 1, None),
    # Third-place playoff
    ("3rd", "England", "France", 6, 4, None),
    # Final
    ("final", "Spain", "Argentina", 1, 0, None),
]

MODEL_TOP15 = [
    "Spain", "Argentina", "France", "England", "Brazil", "Colombia", "Germany",
    "Ecuador", "Mexico", "Portugal", "Netherlands", "Norway", "Turkey", "Morocco", "Japan",
]


def outcome_label(score_a: int, score_b: int) -> str:
    if score_a > score_b:
        return "H"
    if score_a < score_b:
        return "A"
    return "D"


def brier_3way(p: dict, actual: str) -> float:
    y = {"H": 1.0 if actual == "H" else 0.0,
         "D": 1.0 if actual == "D" else 0.0,
         "A": 1.0 if actual == "A" else 0.0}
    return sum((p[k] - y[k]) ** 2 for k in ("H", "D", "A"))


def main() -> None:
    print("Loading match history (2026 WC rows are NA-score, auto-dropped)...")
    results = load_results()
    print(f"  {len(results):,} matches, most recent: {results.date.max().date()}")

    print("\nRecomputing Elo + outcome model exactly as report.py did...")
    ratings, snapshots = compute_elo_history(results, conmebol_qualifier_k_factor=0.5)
    model = fit_outcome_model(snapshots, min_date="2006-01-01")

    # -----------------------------------------------------------------
    # Group stage: 3-way Brier score + pick accuracy
    # -----------------------------------------------------------------
    print("\n" + "=" * 70)
    print("GROUP STAGE (72 matches)")
    print("=" * 70)

    gs_rows = []
    for team_a, team_b, sa, sb in GROUP_STAGE:
        elo_a = ratings[team_a] + (HOST_BOOST if team_a in HOST_TEAMS_2026 else 0.0)
        elo_b = ratings[team_b] + (HOST_BOOST if team_b in HOST_TEAMS_2026 else 0.0)
        p = match_proba(model, elo_a, elo_b, neutral=True)
        actual = outcome_label(sa, sb)
        predicted = max(p, key=p.get)
        gs_rows.append({
            "team_a": team_a, "team_b": team_b, "score": f"{sa}-{sb}",
            "p_h": p["H"], "p_d": p["D"], "p_a": p["A"],
            "actual": actual, "predicted": predicted,
            "hit": actual == predicted,
            "brier": brier_3way(p, actual),
        })
    gs_df = pd.DataFrame(gs_rows)
    print(f"  Pick accuracy (most-likely outcome correct): "
          f"{gs_df['hit'].mean():.1%} ({gs_df['hit'].sum()}/{len(gs_df)})")
    print(f"  Mean 3-way Brier score: {gs_df['brier'].mean():.3f} "
          f"(0=perfect, 0.667=uniform-random guess)")
    draws_actual = (gs_df["actual"] == "D").sum()
    draws_predicted_as_favorite = (gs_df["predicted"] == "D").sum()
    print(f"  Actual draws: {draws_actual}/72 ({draws_actual/72:.1%}). "
          f"Model called draw as most-likely outcome in {draws_predicted_as_favorite} matches.")

    # -----------------------------------------------------------------
    # Knockouts: binary advance Brier score + pick accuracy
    # -----------------------------------------------------------------
    print("\n" + "=" * 70)
    print("KNOCKOUT STAGE (32 matches: 16 R32 + 8 R16 + 4 QF + 2 SF + 3rd + final)")
    print("=" * 70)

    ko_rows = []
    for stage, team_a, team_b, sa, sb, shootout_winner in KNOCKOUTS:
        p_a_advance = knockout_win_proba(model, ratings[team_a], ratings[team_b], neutral=True)
        if shootout_winner is not None:
            actual_a_advances = 1 if shootout_winner == team_a else 0
        else:
            actual_a_advances = 1 if sa > sb else 0
        predicted_a = p_a_advance >= 0.5
        ko_rows.append({
            "stage": stage, "team_a": team_a, "team_b": team_b,
            "score": f"{sa}-{sb}" + (f" ({shootout_winner} on pens)" if shootout_winner else ""),
            "p_a_advance": p_a_advance,
            "actual_a_advances": actual_a_advances,
            "hit": predicted_a == bool(actual_a_advances),
            "brier": (p_a_advance - actual_a_advances) ** 2,
        })
    ko_df = pd.DataFrame(ko_rows)
    print(f"  Pick accuracy (favorite advanced): {ko_df['hit'].mean():.1%} "
          f"({ko_df['hit'].sum()}/{len(ko_df)})")
    print(f"  Mean Brier score: {ko_df['brier'].mean():.3f} (0=perfect, 0.25=coin flip)")

    print("\n  All misses (model's favorite got eliminated), by confidence:")
    misses = ko_df[~ko_df["hit"]].copy()
    misses["fav_p"] = misses["p_a_advance"].clip(lower=1 - misses["p_a_advance"])
    misses = misses.sort_values("fav_p", ascending=False)
    for _, r in misses.iterrows():
        fav, dog = (r["team_a"], r["team_b"]) if r["p_a_advance"] >= 0.5 else (r["team_b"], r["team_a"])
        print(f"    [{r['stage'].upper()}] {dog} eliminated {fav} "
              f"(model gave {fav} {r['fav_p']:.0%} to advance) -- {r['score']}")

    # -----------------------------------------------------------------
    # Final standings vs model's predicted top 15
    # -----------------------------------------------------------------
    print("\n" + "=" * 70)
    print("MODEL'S PREDICTED TOP 15 -- FINAL EXIT ROUND")
    print("=" * 70)

    # Round reached where each team's tournament ended. "3rd" beats "SF" as
    # a placement (it's a later match), so process KNOCKOUTS in stage order
    # and let later entries overwrite earlier ones.
    STAGE_ORDER = ["r32", "r16", "qf", "sf", "3rd", "final"]
    STAGE_LABEL = {
        "r32": "eliminated in R32", "r16": "eliminated in R16",
        "qf": "eliminated in QF", "sf": "eliminated in SF (lost 3rd-place)",
        "3rd": "4th place (lost 3rd-place playoff)",
        "final": "RUNNER-UP (lost final)",
    }

    exit_status: dict[str, str] = {}
    champion = None
    for stage in STAGE_ORDER:
        for s, team_a, team_b, sa, sb, sw in KNOCKOUTS:
            if s != stage:
                continue
            winner = (sw if sw else (team_a if sa > sb else team_b))
            loser = team_b if winner == team_a else team_a
            if stage == "final":
                champion = winner
                exit_status[loser] = "RUNNER-UP (lost final)"
            elif stage == "3rd":
                exit_status[loser] = "4th place (lost 3rd-place playoff)"
                exit_status[winner] = "3RD PLACE (won 3rd-place playoff)"
            else:
                exit_status[loser] = STAGE_LABEL[stage]

    ko_participants = {t for _, a, b, *_ in KNOCKOUTS for t in (a, b)}

    for rank, team in enumerate(MODEL_TOP15, start=1):
        if team == champion:
            status = "CHAMPION"
        elif team in exit_status:
            status = exit_status[team]
        elif team in ko_participants:
            status = "eliminated in R32"  # lost their only listed knockout game
        else:
            status = "eliminated in GROUP STAGE"
        print(f"  #{rank:<3}{team:<14} {status}")

    # -----------------------------------------------------------------
    # Headline verdict: where did the actual champion/finalists rank?
    # -----------------------------------------------------------------
    print("\n" + "=" * 70)
    print("HEADLINE VERDICT")
    print("=" * 70)
    runner_up = next(t for t, s in exit_status.items() if s.startswith("RUNNER-UP"))
    third = next(t for t, s in exit_status.items() if s.startswith("3RD PLACE"))
    fourth = next(t for t, s in exit_status.items() if s.startswith("4th place"))

    def rank_of(team: str) -> str:
        return f"#{MODEL_TOP15.index(team) + 1}" if team in MODEL_TOP15 else "unranked (outside top 15)"

    print(f"  Actual champion : {champion:<12} (model rank {rank_of(champion)})")
    print(f"  Actual runner-up: {runner_up:<12} (model rank {rank_of(runner_up)})")
    print(f"  Actual 3rd place: {third:<12} (model rank {rank_of(third)})")
    print(f"  Actual 4th place: {fourth:<12} (model rank {rank_of(fourth)})")


if __name__ == "__main__":
    main()
