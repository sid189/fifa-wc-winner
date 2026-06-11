"""Squad market value loader + Elo augmentation.

Squad value (Transfermarkt total) is a team-strength signal Elo can't see:
peak-form squads with depressed Elo (Italy 2006, Brazil 2002 cases).

The augmentation is:
    augmented_elo[team] = elo[team] + alpha * (log_value - mean_log_value)

So teams above the mean log value get bumped up; below the mean get bumped
down. The default `alpha=30` gives roughly +/- 50 Elo of swing across a typical
WC field, which is large enough to matter without dominating the Elo signal.
"""
from __future__ import annotations

import math
from pathlib import Path

import pandas as pd

CSV_PATH = Path(__file__).resolve().parent.parent / "data" / "squad_values.csv"
DEFAULT_ALPHA = 30.0


def load_squad_values() -> pd.DataFrame:
    if not CSV_PATH.exists():
        return pd.DataFrame(columns=["year", "team", "squad_value_eur_m"])
    df = pd.read_csv(CSV_PATH)
    df["year"] = df["year"].astype(int)
    df["squad_value_eur_m"] = df["squad_value_eur_m"].astype(float)
    return df


def get_squad_value(team: str, year: int, df: pd.DataFrame) -> float | None:
    """Return the squad value for `team` closest to `year`, or None."""
    sub = df[df["team"] == team]
    if sub.empty:
        return None
    closest = sub.iloc[(sub["year"] - year).abs().argsort().iloc[0]]
    return float(closest["squad_value_eur_m"])


def augment_elo(
    ratings: dict[str, float],
    year: int,
    teams: list[str] | None = None,
    alpha: float = DEFAULT_ALPHA,
    df: pd.DataFrame | None = None,
) -> dict[str, float]:
    """Return new ratings dict with squad-value augmentation.

    `teams` restricts mean-log calculation to a specific field (e.g. the 48
    qualified teams for that year), so the augmentation is centered on that
    cohort rather than the entire Elo table.
    """
    df = df if df is not None else load_squad_values()
    if df.empty:
        return dict(ratings)

    cohort = teams if teams is not None else list(ratings.keys())
    log_values: dict[str, float] = {}
    for team in cohort:
        v = get_squad_value(team, year, df)
        if v is not None and v > 0:
            log_values[team] = math.log(v)

    if not log_values:
        return dict(ratings)

    mean_log = sum(log_values.values()) / len(log_values)
    out = dict(ratings)
    for team, lv in log_values.items():
        if team in out:
            out[team] = out[team] + alpha * (lv - mean_log)
    return out
