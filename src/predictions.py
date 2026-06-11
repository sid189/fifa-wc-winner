"""Match-level probability tables (group stage and knockout)."""
from __future__ import annotations

from itertools import combinations
import pandas as pd

from .model import OutcomeModel, match_proba, knockout_win_proba
from .simulate import HOST_BOOST


def group_stage_predictions(
    groups: dict[str, list[str]],
    ratings: dict[str, float],
    model: OutcomeModel,
    host_teams: set[str] | None = None,
) -> pd.DataFrame:
    """Win/draw/loss probability for every group-stage match.

    All 12 groups have 6 matches each (round-robin), so this returns
    72 rows for the full 2026 group stage.
    """
    hosts = host_teams or set()
    rows = []
    for gname, teams in groups.items():
        for a, b in combinations(teams, 2):
            elo_a = ratings[a] + (HOST_BOOST if a in hosts else 0.0)
            elo_b = ratings[b] + (HOST_BOOST if b in hosts else 0.0)
            p = match_proba(model, elo_a, elo_b, neutral=True)
            most_likely = max(("win", p["H"]), ("draw", p["D"]), ("loss", p["A"]),
                              key=lambda kv: kv[1])
            if most_likely[0] == "win":
                forecast = f"{a} win"
            elif most_likely[0] == "loss":
                forecast = f"{b} win"
            else:
                forecast = "Draw"
            rows.append({
                "group": gname,
                "team_a": a,
                "team_b": b,
                "p_a_win": p["H"],
                "p_draw": p["D"],
                "p_b_win": p["A"],
                "forecast": forecast,
                "confidence": most_likely[1],
            })
    return pd.DataFrame(rows)


def format_group_stage_table(preds: pd.DataFrame) -> str:
    """Pretty-print group-stage predictions grouped by group letter."""
    out = []
    for gname, sub in preds.groupby("group"):
        out.append(f"\n=== GROUP {gname} ===")
        for _, r in sub.iterrows():
            tag = "  [DRAW]" if r["forecast"] == "Draw" else ""
            out.append(
                f"  {r['team_a']:>22} vs {r['team_b']:<22} "
                f"  Win={r['p_a_win']:.2f}  Draw={r['p_draw']:.2f}  Loss={r['p_b_win']:.2f}"
                f"  ->  {r['forecast']} ({r['confidence']:.0%}){tag}"
            )
    return "\n".join(out)


def draw_watch(preds: pd.DataFrame, top_n: int = 8) -> pd.DataFrame:
    """Matches most likely to end in a draw."""
    return (
        preds.sort_values("p_draw", ascending=False)
        .head(top_n)[["group", "team_a", "team_b", "p_draw", "p_a_win", "p_b_win"]]
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# R32 (deterministic "most likely seeding") predictions
# ---------------------------------------------------------------------------

def most_likely_seeding(
    groups: dict[str, list[str]],
    ratings: dict[str, float],
    host_teams: set[str] | None = None,
) -> tuple[dict[str, str], dict[str, str], list[tuple[str, str]]]:
    """Return (firsts, seconds, thirds) if every group goes to chalk.

    "Chalk" = order by host-adjusted Elo within the group, then take the
    best-8 third-placed teams by raw Elo. Useful for showing a single
    representative R32 matchup table; doesn't capture upset risk.
    """
    hosts = host_teams or set()
    firsts: dict[str, str] = {}
    seconds: dict[str, str] = {}
    thirds_per_group: list[tuple[str, str, float]] = []  # (team, group, elo)

    for gname, teams in groups.items():
        ranked = sorted(
            teams,
            key=lambda t: -(ratings[t] + (HOST_BOOST if t in hosts else 0.0)),
        )
        firsts[gname] = ranked[0]
        seconds[gname] = ranked[1]
        thirds_per_group.append((ranked[2], gname, ratings[ranked[2]]))

    # Best 8 third-placed by raw Elo (proxy for the FIFA tiebreaker).
    thirds_per_group.sort(key=lambda x: -x[2])
    thirds = [(t[0], t[1]) for t in thirds_per_group[:8]]
    return firsts, seconds, thirds


def r32_predictions(
    bracket: list[str],
    ratings: dict[str, float],
    model: OutcomeModel,
) -> pd.DataFrame:
    """Match-by-match win probability for the 16 R32 fixtures.

    Knockouts treated as neutral (host advantage stops after groups).
    Shows both 90-min H/D/A and the advancement probability (draws -> pens
    split 50/50).
    """
    rows = []
    for i in range(0, len(bracket), 2):
        a, b = bracket[i], bracket[i + 1]
        p = match_proba(model, ratings[a], ratings[b], neutral=True)
        adv_a = knockout_win_proba(model, ratings[a], ratings[b], neutral=True)
        rows.append({
            "match": f"R32-{i // 2 + 1}",
            "team_a": a,
            "team_b": b,
            "p_a_win_90": p["H"],
            "p_draw_90": p["D"],
            "p_b_win_90": p["A"],
            "p_a_advance": adv_a,
            "p_b_advance": 1 - adv_a,
            "favorite": a if adv_a >= 0.5 else b,
            "fav_advance": max(adv_a, 1 - adv_a),
        })
    return pd.DataFrame(rows)


def format_r32_table(preds: pd.DataFrame) -> str:
    out = ["", "=== ROUND OF 32 - most-likely-seeding view ==="]
    for _, r in preds.iterrows():
        out.append(
            f"  {r['match']}: {r['team_a']:>22} vs {r['team_b']:<22}"
            f"  90'(W/D/L)={r['p_a_win_90']:.2f}/{r['p_draw_90']:.2f}/{r['p_b_win_90']:.2f}"
            f"   Advance: {r['favorite']} {r['fav_advance']:.0%}"
        )
    return "\n".join(out)


# ---------------------------------------------------------------------------
# Full chalk-path bracket: R32 -> R16 -> QF -> SF -> Final + Third Place
# ---------------------------------------------------------------------------

_ROUND_TITLES = {
    "R32": "ROUND OF 32",
    "R16": "ROUND OF 16",
    "QF":  "QUARTERFINALS",
    "SF":  "SEMIFINALS",
    "F":   "FINAL",
    "3rd": "THIRD-PLACE PLAYOFF",
}


def _ko_row(round_name: str, match_label: str, a: str, b: str,
            ratings: dict, model: OutcomeModel) -> tuple[dict, str, str]:
    """Compute a single KO row + return (row, winner, loser) under chalk."""
    p = match_proba(model, ratings[a], ratings[b], neutral=True)
    adv_a = knockout_win_proba(model, ratings[a], ratings[b], neutral=True)
    winner, loser = (a, b) if adv_a >= 0.5 else (b, a)
    row = {
        "round": round_name,
        "match": match_label,
        "team_a": a,
        "team_b": b,
        "p_a_win_90": p["H"],
        "p_draw_90": p["D"],
        "p_b_win_90": p["A"],
        "p_a_advance": adv_a,
        "p_b_advance": 1 - adv_a,
        "favorite": winner,
        "fav_advance": max(adv_a, 1 - adv_a),
    }
    return row, winner, loser


def full_bracket_predictions(
    bracket: list[str],
    ratings: dict[str, float],
    model: OutcomeModel,
) -> pd.DataFrame:
    """Walk the bracket along the chalk path (favorite advances each round).

    Returns every KO match from R32 through Final, plus the Third-Place
    playoff between the two SF losers. This is a single deterministic path
    - use the Monte Carlo simulation for actual win probabilities, this view
    just shows what each round looks like IF every favorite holds.
    """
    rows: list[dict] = []
    sf_losers: list[str] = []

    rounds = ["R32", "R16", "QF", "SF", "F"]
    current = list(bracket)

    for round_name in rounds:
        nxt: list[str] = []
        for i in range(0, len(current), 2):
            a, b = current[i], current[i + 1]
            label = f"{round_name}-{i // 2 + 1}"
            row, winner, loser = _ko_row(round_name, label, a, b, ratings, model)
            rows.append(row)
            if round_name == "SF":
                sf_losers.append(loser)
            nxt.append(winner)
        current = nxt

    if len(sf_losers) == 2:
        a, b = sf_losers
        row, _, _ = _ko_row("3rd", "3rd Place", a, b, ratings, model)
        rows.append(row)

    return pd.DataFrame(rows)


def format_full_bracket_table(preds: pd.DataFrame) -> str:
    """Pretty-print the full chalk bracket grouped by round."""
    out: list[str] = []
    for round_name, group in preds.groupby("round", sort=False):
        out.append(f"\n=== {_ROUND_TITLES.get(round_name, round_name)} ===")
        for _, r in group.iterrows():
            out.append(
                f"  {r['match']:>9}: {r['team_a']:>22} vs {r['team_b']:<22}"
                f"  90'(W/D/L)={r['p_a_win_90']:.2f}/{r['p_draw_90']:.2f}/{r['p_b_win_90']:.2f}"
                f"   Advance: {r['favorite']} {r['fav_advance']:.0%}"
            )
    return "\n".join(out)
