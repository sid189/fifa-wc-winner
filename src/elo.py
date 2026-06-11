"""World Football Elo (eloratings.net convention).

Reference: https://www.eloratings.net/about
- K depends on tournament weight.
- Home advantage = 100 Elo points (skipped on neutral grounds).
- Margin-of-victory multiplier G expands K when matches are blowouts.
"""
from __future__ import annotations

from collections import defaultdict
import math
import pandas as pd
from tqdm import tqdm

DEFAULT_RATING = 1500.0
HOME_ADV = 100.0

# 10 CONMEBOL members. Used for the qualifier-K override experiment that tests
# whether CONMEBOL's brutal round-robin qualification format unfairly deflates
# big-team Elo (Brazil-2002 was rank #9 in the model; suspected fix.)
CONMEBOL_TEAMS: set[str] = {
    "Argentina", "Bolivia", "Brazil", "Chile", "Colombia",
    "Ecuador", "Paraguay", "Peru", "Uruguay", "Venezuela",
}

# Tournament weight (K). Keys are matched as case-insensitive substrings.
K_BY_TOURNAMENT = {
    "fifa world cup": 60,
    "world cup qualification": 50,
    "uefa euro": 50,
    "copa américa": 50,
    "copa america": 50,
    "african cup of nations": 50,
    "africa cup of nations": 50,
    "afc asian cup": 50,
    "confederations cup": 40,
    "nations league": 40,
    "uefa nations": 40,
    "friendly": 20,
}
K_DEFAULT = 30


def k_for(tournament: str) -> int:
    t = (tournament or "").lower()
    for needle, k in K_BY_TOURNAMENT.items():
        if needle in t:
            return k
    return K_DEFAULT


def mov_multiplier(goal_diff: int) -> float:
    g = abs(goal_diff)
    if g <= 1:
        return 1.0
    if g == 2:
        return 1.5
    return (11 + g) / 8.0


def expected(rating_a: float, rating_b: float) -> float:
    return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))


def compute_elo_history(
    results: pd.DataFrame,
    conmebol_qualifier_k_factor: float = 1.0,
) -> tuple[dict[str, float], pd.DataFrame]:
    """Run Elo over the full match history.

    Args:
        conmebol_qualifier_k_factor: multiplier on K for CONMEBOL-vs-CONMEBOL
            qualification matches. 1.0 = default (no change); 0.5 = halve
            the K factor to test whether CONMEBOL Elo is depressed by its
            qualification format. (See backtest_all.py — Brazil-2002 puzzle.)

    Returns:
        ratings: final rating per team
        snapshots: per-match rows with pre-match ratings (training features)
    """
    ratings: dict[str, float] = defaultdict(lambda: DEFAULT_RATING)
    rows = []
    df = results.sort_values("date").reset_index(drop=True)

    for r in tqdm(df.itertuples(index=False), total=len(df), desc="Elo"):
        home, away = r.home_team, r.away_team
        neutral = bool(r.neutral)
        r_home = ratings[home]
        r_away = ratings[away]

        adj_home = r_home + (0.0 if neutral else HOME_ADV)
        e_home = expected(adj_home, r_away)

        gd = int(r.home_score) - int(r.away_score)
        if gd > 0:
            s_home = 1.0
        elif gd < 0:
            s_home = 0.0
        else:
            s_home = 0.5

        k = k_for(r.tournament) * mov_multiplier(gd)
        if (conmebol_qualifier_k_factor != 1.0
                and "qualification" in (r.tournament or "").lower()
                and home in CONMEBOL_TEAMS and away in CONMEBOL_TEAMS):
            k *= conmebol_qualifier_k_factor
        delta = k * (s_home - e_home)

        rows.append(
            {
                "date": r.date,
                "tournament": r.tournament,
                "home_team": home,
                "away_team": away,
                "neutral": neutral,
                "home_score": int(r.home_score),
                "away_score": int(r.away_score),
                "home_elo_pre": r_home,
                "away_elo_pre": r_away,
                "elo_diff": adj_home - r_away,
                "outcome": "H" if gd > 0 else ("A" if gd < 0 else "D"),
            }
        )

        ratings[home] = r_home + delta
        ratings[away] = r_away - delta

    snapshots = pd.DataFrame(rows)
    return dict(ratings), snapshots


def top_n(ratings: dict[str, float], n: int = 25) -> pd.DataFrame:
    s = pd.Series(ratings, name="elo").sort_values(ascending=False)
    return s.head(n).reset_index().rename(columns={"index": "team"})
