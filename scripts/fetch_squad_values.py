"""Fetch current squad market values for 2026 WC teams via the TM API.

Appends to data/squad_values.csv. Skips teams already present for year 2026.
The API only exposes CURRENT values, so this script only populates 2026 rows -
historical squad values for backtest years (1998-2022) must be added manually
from public reporting (Al Jazeera, CIES Football Observatory, Boardroom, etc).

Run:
    python scripts/fetch_squad_values.py
"""
from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.groups_2026 import GROUPS_2026

API = "https://transfermarkt-api.fly.dev"
CSV_PATH = ROOT / "data" / "squad_values.csv"
DELAY = 0.6  # seconds between API calls
YEAR = 2026

# Name overrides for teams whose TM name differs from our dataset name.
TM_NAME_OVERRIDES: dict[str, str] = {
    "South Korea": "Korea, South",
    "Ivory Coast": "Cote d'Ivoire",
    "Cape Verde": "Cape Verde",
    "DR Congo": "DR Congo",
    "Czech Republic": "Czech Republic",
    "Turkey": "Türkiye",
    "Curaçao": "Curacao",
    "Bosnia and Herzegovina": "Bosnia-Herzegovina",
}


def fetch_value(team_name: str) -> float | None:
    """Return current squad market value in EUR millions, or None if not found."""
    query = TM_NAME_OVERRIDES.get(team_name, team_name)
    url = f"{API}/clubs/search/{query}"
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    results = r.json().get("results", [])
    # Prefer exact name + country match (rules out U20/U17 / club teams with similar names).
    for x in results:
        if x.get("name") == query and x.get("country") == query:
            return (x.get("marketValue") or 0) / 1_000_000
    # Fallback: first result if no exact match.
    if results:
        return (results[0].get("marketValue") or 0) / 1_000_000
    return None


def _load_existing() -> set[tuple[int, str]]:
    if not CSV_PATH.exists():
        return set()
    out = set()
    with open(CSV_PATH) as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                out.add((int(row["year"]), row["team"]))
            except (KeyError, ValueError):
                pass
    return out


def main() -> None:
    existing = _load_existing()
    CSV_PATH.parent.mkdir(exist_ok=True)
    write_header = not CSV_PATH.exists()

    teams = [t for grp in GROUPS_2026.values() for t in grp]
    missing = [t for t in teams if (YEAR, t) not in existing]
    print(f"{len(missing)} teams to fetch for {YEAR} "
          f"({len(teams) - len(missing)} already cached).")

    with open(CSV_PATH, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["year", "team", "squad_value_eur_m"])
        for i, team in enumerate(missing, 1):
            try:
                value = fetch_value(team)
                if value is None:
                    print(f"  [{i}/{len(missing)}] {team}: no API result")
                    continue
                print(f"  [{i}/{len(missing)}] {team}: €{value:.0f}m")
                writer.writerow([YEAR, team, f"{value:.1f}"])
                f.flush()
            except Exception as e:
                print(f"  [{i}/{len(missing)}] {team}: error {e}")
            time.sleep(DELAY)

    print(f"\nWrote to {CSV_PATH}")


if __name__ == "__main__":
    main()
