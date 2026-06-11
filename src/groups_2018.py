"""Actual 2018 FIFA World Cup groups (Russia).

Reuses `bracket_2022` from groups_2022.py since the 32-team FIFA bracket
pattern hasn't changed.
"""
from __future__ import annotations

from .groups_2022 import bracket_2022 as bracket_2018  # noqa: F401

GROUPS_2018: dict[str, list[str]] = {
    "A": ["Russia", "Saudi Arabia", "Egypt", "Uruguay"],
    "B": ["Portugal", "Spain", "Morocco", "Iran"],
    "C": ["France", "Australia", "Peru", "Denmark"],
    "D": ["Argentina", "Iceland", "Croatia", "Nigeria"],
    "E": ["Brazil", "Switzerland", "Costa Rica", "Serbia"],
    "F": ["Germany", "Mexico", "Sweden", "South Korea"],
    "G": ["Belgium", "Panama", "Tunisia", "England"],
    "H": ["Poland", "Senegal", "Colombia", "Japan"],
}

ACTUAL_2018 = {
    "Champion": "France",
    "Runner-up": "Croatia",
    "Semi-finalists": ["Belgium", "England"],
}
