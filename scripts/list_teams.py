"""Print recent international-side team names so you can match GROUPS_2026."""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import load_results


def main() -> None:
    df = load_results()
    recent = df[df["date"] >= "2023-01-01"]
    teams = sorted(set(recent["home_team"]).union(recent["away_team"]))
    for t in teams:
        print(t)


if __name__ == "__main__":
    main()
