"""Monte Carlo simulator for World Cup tournaments.

Generalised to handle both formats:
- 2022 / earlier: 8 groups of 4, top 2 advance to R16 (no third-place qualifiers).
- 2026: 12 groups of 4, top 2 + 8 best third-placed teams advance to R32.

`bracket_fn` lets the caller plug in a real seeding (e.g. FIFA's 1A-2B pattern)
instead of the default random R16/R32 draw.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from typing import Callable, Optional
import numpy as np
import pandas as pd
from tqdm import tqdm

from .model import OutcomeModel, match_proba, knockout_win_proba

# bracket_fn(firsts, seconds, thirds) -> ordered list of 2^k teams.
#   firsts:  dict[group_letter -> team]
#   seconds: dict[group_letter -> team]
#   thirds:  list[(team, source_group)] for the qualifying third-placed teams
BracketFn = Callable[[dict, dict, list], list[str]]
HOST_BOOST = 100.0  # Elo points added to a host playing in its own country


def _boost(team: str, hosts: set[str] | None) -> float:
    return HOST_BOOST if hosts and team in hosts else 0.0


def _sim_group(group: list[str], ratings: dict, model: OutcomeModel,
               rng: np.random.Generator, hosts: set[str] | None = None):
    pts = defaultdict(int)
    gd = defaultdict(int)
    for i in range(len(group)):
        for j in range(i + 1, len(group)):
            a, b = group[i], group[j]
            # Host nations play group games at home -> bake the boost into Elo.
            elo_a = ratings[a] + _boost(a, hosts)
            elo_b = ratings[b] + _boost(b, hosts)
            p = match_proba(model, elo_a, elo_b, neutral=True)
            r = rng.random()
            if r < p["H"]:
                pts[a] += 3; gd[a] += 1; gd[b] -= 1
            elif r < p["H"] + p["D"]:
                pts[a] += 1; pts[b] += 1
            else:
                pts[b] += 3; gd[b] += 1; gd[a] -= 1
    standings = sorted(group, key=lambda t: (pts[t], gd[t], rng.random()), reverse=True)
    return standings, pts, gd


def _ko(model: OutcomeModel, ratings: dict, a: str, b: str, rng: np.random.Generator) -> str:
    p = knockout_win_proba(model, ratings[a], ratings[b], neutral=True)
    return a if rng.random() < p else b


def _stage_names_for(n_teams: int) -> list[str]:
    rename = {1: "Champion", 2: "F", 4: "SF", 8: "QF"}
    out = []
    n = n_teams
    while n > 1:
        n //= 2
        out.append(rename.get(n, f"R{n}"))
    return out


def simulate_tournament(
    groups: dict[str, list[str]],
    ratings: dict[str, float],
    model: OutcomeModel,
    rng: np.random.Generator,
    n_third_qualifiers: int = 8,
    bracket_fn: Optional[BracketFn] = None,
    host_teams: Optional[set[str]] = None,
) -> dict:
    """Play one full tournament.

    n_third_qualifiers: 8 for 2026, 0 for 2022/2018.
    bracket_fn: f(firsts, seconds, thirds) -> ordered list of 2^k teams.
                Receives dicts keyed by group name. Defaults to random shuffle.
    host_teams: nations that play group games at home (e.g. {"United States",
                "Mexico", "Canada"} for 2026). Gets +100 Elo on those games.
                Knockouts are treated as neutral (host doesn't always play in
                its own country once the bracket starts).
    """
    firsts: dict[str, str] = {}
    seconds: dict[str, str] = {}
    third_pool = []

    for gname, teams in groups.items():
        standings, pts, gd = _sim_group(teams, ratings, model, rng, hosts=host_teams)
        firsts[gname] = standings[0]
        seconds[gname] = standings[1]
        third_pool.append((standings[2], pts[standings[2]], gd[standings[2]], gname))

    third_pool.sort(key=lambda x: (x[1], x[2], rng.random()), reverse=True)
    thirds_with_groups = [(t[0], t[3]) for t in third_pool[:n_third_qualifiers]]
    seeds_3rd_teams = [t[0] for t in thirds_with_groups]

    if bracket_fn is not None:
        bracket = bracket_fn(firsts, seconds, thirds_with_groups)
    else:
        bracket = list(firsts.values()) + list(seconds.values()) + seeds_3rd_teams
        rng.shuffle(bracket)

    stage_reached = {t: f"R{len(bracket)}" for t in bracket}
    for stage in _stage_names_for(len(bracket)):
        nxt = []
        for i in range(0, len(bracket), 2):
            w = _ko(model, ratings, bracket[i], bracket[i + 1], rng)
            stage_reached[w] = stage
            nxt.append(w)
        bracket = nxt

    return {"winner": bracket[0], "stage_reached": stage_reached}


def monte_carlo(
    groups: dict[str, list[str]],
    ratings: dict[str, float],
    model: OutcomeModel,
    n_sims: int = 10_000,
    seed: int = 42,
    n_third_qualifiers: int = 8,
    bracket_fn: Optional[BracketFn] = None,
    host_teams: Optional[set[str]] = None,
) -> pd.DataFrame:
    missing = [t for grp in groups.values() for t in grp if t not in ratings]
    if missing:
        raise KeyError(
            f"Teams missing from Elo table: {missing}. "
            "Run `python scripts/list_teams.py | grep -i <name>` to find the correct spelling."
        )
    rng = np.random.default_rng(seed)

    # Track each team's *final* stage per sim. We aggregate cumulatives at the end.
    STAGES = ["R32", "R16", "QF", "SF", "F", "Champion"]
    rank_of = {s: i for i, s in enumerate(STAGES)}
    final_stage: dict[str, Counter] = {s: Counter() for s in STAGES}

    for _ in tqdm(range(n_sims), desc="Monte Carlo"):
        res = simulate_tournament(
            groups, ratings, model, rng,
            n_third_qualifiers=n_third_qualifiers,
            bracket_fn=bracket_fn,
            host_teams=host_teams,
        )
        for team, st in res["stage_reached"].items():
            if st in final_stage:
                final_stage[st][team] += 1

    def cumulative(team: str, stage: str) -> int:
        floor = rank_of[stage]
        return sum(final_stage[s][team] for s in STAGES if rank_of[s] >= floor)

    teams = [t for grp in groups.values() for t in grp]
    rows = [
        {
            "team": t,
            "elo": ratings.get(t, np.nan),
            "p_r32": cumulative(t, "R32") / n_sims,
            "p_r16": cumulative(t, "R16") / n_sims,
            "p_qf": cumulative(t, "QF") / n_sims,
            "p_semi": cumulative(t, "SF") / n_sims,
            "p_final": cumulative(t, "F") / n_sims,
            "p_champion": cumulative(t, "Champion") / n_sims,
        }
        for t in teams
    ]
    return (
        pd.DataFrame(rows)
        .sort_values("p_champion", ascending=False)
        .reset_index(drop=True)
    )
