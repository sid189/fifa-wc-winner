"""Fetch match-results data from martj42/international_results (GitHub)."""
from __future__ import annotations

from pathlib import Path
import pandas as pd
import requests

RESULTS_URL = (
    "https://raw.githubusercontent.com/martj42/"
    "international_results/master/results.csv"
)
SHOOTOUTS_URL = (
    "https://raw.githubusercontent.com/martj42/"
    "international_results/master/shootouts.csv"
)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def _download(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    dest.write_bytes(resp.content)
    return dest


def load_results(refresh: bool = False) -> pd.DataFrame:
    path = DATA_DIR / "results.csv"
    if refresh or not path.exists():
        _download(RESULTS_URL, path)
    df = pd.read_csv(path, parse_dates=["date"])
    df["home_score"] = df["home_score"].astype("Int64")
    df["away_score"] = df["away_score"].astype("Int64")
    return df.dropna(subset=["home_score", "away_score"]).reset_index(drop=True)


def load_shootouts(refresh: bool = False) -> pd.DataFrame:
    path = DATA_DIR / "shootouts.csv"
    if refresh or not path.exists():
        _download(SHOOTOUTS_URL, path)
    return pd.read_csv(path, parse_dates=["date"])
