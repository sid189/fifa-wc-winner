"""Visualization helpers for World Cup predictions."""
from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd

from .model import OutcomeModel, knockout_win_proba

STAGES = ["R32", "R16", "QF", "SF", "F", "Champion"]
STAGE_LABELS = {
    "R32": "Reach R32",
    "R16": "Reach R16",
    "QF": "Reach QF",
    "SF": "Reach SF",
    "F": "Reach Final",
    "Champion": "Win it all",
}


def plot_funnel(preds: pd.DataFrame, top_n: int = 10, ax=None):
    """Stacked stage probabilities for the top N contenders.

    Shows how each team's championship probability is built up: how often
    they clear groups, then R16, then QF, etc. The visual "fall-off" is
    where each team typically gets eliminated.
    """
    df = preds.head(top_n).copy()
    col_for = {"R32": "p_r32", "R16": "p_r16", "QF": "p_qf",
               "SF": "p_semi", "F": "p_final", "Champion": "p_champion"}
    stages = [s for s in STAGES if df[col_for[s]].max() > 0]

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))
    width = 0.85 / len(stages)
    x = np.arange(len(df))
    for i, s in enumerate(stages):
        ax.bar(x + i * width, df[col_for[s]], width, label=STAGE_LABELS[s])
    ax.set_xticks(x + width * (len(stages) - 1) / 2)
    ax.set_xticklabels(df["team"], rotation=35, ha="right")
    ax.set_ylabel("Probability")
    ax.set_ylim(0, 1.0)
    ax.set_title(f"Tournament progression - top {top_n} contenders")
    ax.legend(ncol=3, fontsize=9, loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    return ax


def plot_h2h_heatmap(teams: list[str], ratings: dict, model: OutcomeModel, ax=None):
    """Pairwise knockout win probability (P(row beats column) on neutral ground)."""
    n = len(teams)
    m = np.full((n, n), np.nan)
    for i, a in enumerate(teams):
        for j, b in enumerate(teams):
            if i == j:
                continue
            m[i, j] = knockout_win_proba(model, ratings[a], ratings[b], neutral=True)

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(m, cmap="RdYlGn", vmin=0, vmax=1)
    ax.set_xticks(range(n)); ax.set_xticklabels(teams, rotation=45, ha="right")
    ax.set_yticks(range(n)); ax.set_yticklabels(teams)
    for i in range(n):
        for j in range(n):
            if i != j:
                ax.text(j, i, f"{m[i, j]:.2f}", ha="center", va="center",
                        fontsize=8, color="black")
    plt.colorbar(im, ax=ax, label="P(row beats column)")
    ax.set_title("Knockout win probability (neutral ground)")
    plt.tight_layout()
    return ax


def plot_elo_trajectory(snapshots: pd.DataFrame, teams: list[str],
                        since: str = "2020-01-01", ax=None):
    """Pre-match Elo over time for each team (one line per team)."""
    df = snapshots[snapshots["date"] >= pd.Timestamp(since)]
    home = df[["date", "home_team", "home_elo_pre"]].rename(
        columns={"home_team": "team", "home_elo_pre": "elo"})
    away = df[["date", "away_team", "away_elo_pre"]].rename(
        columns={"away_team": "team", "away_elo_pre": "elo"})
    ts = pd.concat([home, away], ignore_index=True).sort_values("date")

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))
    for t in teams:
        s = ts[ts["team"] == t]
        if s.empty:
            print(f"  no recent matches for {t}")
            continue
        ax.plot(s["date"], s["elo"], label=t, alpha=0.85, linewidth=1.5)
    ax.set_xlabel("Date")
    ax.set_ylabel("Elo (pre-match)")
    ax.set_title(f"Elo trajectory since {since}")
    ax.legend(fontsize=9, ncol=2)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    return ax


def plot_champion_bar(preds: pd.DataFrame, top_n: int = 12, ax=None):
    """Horizontal bar chart of P(champion) for the top N."""
    top = preds.head(top_n).iloc[::-1]
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(top["team"], top["p_champion"], color="steelblue")
    ax.set_xlabel("P(champion)")
    ax.set_title(f"P(champion) - top {top_n}")
    for i, v in enumerate(top["p_champion"]):
        ax.text(v + 0.002, i, f"{v:.1%}", va="center", fontsize=9)
    plt.tight_layout()
    return ax


def plot_chalk_bracket(full_df: pd.DataFrame, ax=None):
    """Draw the 5-round single-elim bracket along the chalk path.

    `full_df` is the DataFrame from `predictions.full_bracket_predictions`.
    Favorites are highlighted in bold; chalk-path winners trace through to the
    Champion box on the right.
    """
    rounds = ["R32", "R16", "QF", "SF", "F"]
    df = full_df[full_df["round"].isin(rounds)].reset_index(drop=True)

    if ax is None:
        fig, ax = plt.subplots(figsize=(16, 11))

    # Y positions: R32 matches at integer y; each subsequent round midpoints.
    positions: dict[tuple[str, int], float] = {}
    r32_sub = df[df["round"] == "R32"].reset_index(drop=True)
    for i in range(len(r32_sub)):
        positions[("R32", i)] = float(i)
    parent_of = {"R16": "R32", "QF": "R16", "SF": "QF", "F": "SF"}
    for r in ["R16", "QF", "SF", "F"]:
        sub = df[df["round"] == r].reset_index(drop=True)
        prev = parent_of[r]
        for i in range(len(sub)):
            y1 = positions[(prev, 2 * i)]
            y2 = positions[(prev, 2 * i + 1)]
            positions[(r, i)] = (y1 + y2) / 2

    box_w, box_h = 0.86, 0.75
    round_to_x = {r: i for i, r in enumerate(rounds)}

    for r in rounds:
        x = round_to_x[r]
        sub = df[df["round"] == r].reset_index(drop=True)
        for i in range(len(sub)):
            y = positions[(r, i)]
            row = sub.iloc[i]
            a, b, fav = row["team_a"], row["team_b"], row["favorite"]
            ax.add_patch(Rectangle(
                (x - box_w / 2, y - box_h / 2), box_w, box_h,
                facecolor="white", edgecolor="0.3", linewidth=0.6,
            ))
            for who, dy in [(a, 0.15), (b, -0.15)]:
                is_fav = who == fav
                ax.text(x, y + dy, who,
                        ha="center", va="center", fontsize=7.5,
                        fontweight="bold" if is_fav else "normal",
                        color="#1a5e1a" if is_fav else "0.25")

    # Connecting lines parent -> child.
    for r in ["R16", "QF", "SF", "F"]:
        prev = parent_of[r]
        sub = df[df["round"] == r].reset_index(drop=True)
        for i in range(len(sub)):
            xc = round_to_x[r]
            xp = round_to_x[prev]
            yp1 = positions[(prev, 2 * i)]
            yp2 = positions[(prev, 2 * i + 1)]
            yc = positions[(r, i)]
            x_mid = (xp + xc) / 2
            ax.plot([xp + box_w / 2, x_mid, x_mid, xc - box_w / 2],
                    [yp1, yp1, yc, yc], color="0.5", linewidth=0.5)
            ax.plot([xp + box_w / 2, x_mid, x_mid, xc - box_w / 2],
                    [yp2, yp2, yc, yc], color="0.5", linewidth=0.5)

    # Champion annotation.
    f_row = df[df["round"] == "F"].iloc[0]
    champ_y = positions[("F", 0)]
    ax.text(round_to_x["F"] + 0.7, champ_y,
            f"Champion:\n{f_row['favorite']}",
            fontsize=12, fontweight="bold", color="#0c4f0c",
            va="center", ha="left")

    n_r32 = len(r32_sub)
    ax.set_xlim(-0.7, len(rounds) - 0.3 + 1.0)
    ax.set_ylim(-1.0, n_r32)
    ax.set_xticks(range(len(rounds)))
    ax.set_xticklabels(["Round of 32", "Round of 16", "Quarterfinals",
                        "Semifinals", "Final"])
    ax.set_yticks([])
    ax.tick_params(axis="x", labelsize=10)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title("Chalk-path bracket - favorite advances each round",
                 fontsize=13, pad=12)
    plt.tight_layout()
    return ax


def plot_confederation_breakdown(preds: pd.DataFrame,
                                 confed_map: dict[str, list[str]],
                                 ax=None):
    """Pie chart of aggregate P(champion) by confederation."""
    team_to_conf = {t: c for c, teams in confed_map.items() for t in teams}
    df = preds.copy()
    df["confederation"] = df["team"].map(team_to_conf).fillna("Other")
    grouped = (df.groupby("confederation")["p_champion"].sum()
               .sort_values(ascending=False))

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 6))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#7f7f7f"]
    labels = [f"{c}\n{p:.1%}" for c, p in grouped.items()]
    ax.pie(grouped.values, labels=labels, colors=colors[:len(grouped)],
           startangle=90, wedgeprops={"edgecolor": "white", "linewidth": 1.5})
    ax.set_title("Aggregate P(champion) by confederation", fontsize=12)
    plt.tight_layout()
    return ax
