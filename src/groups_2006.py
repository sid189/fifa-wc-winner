"""Actual 2006 FIFA World Cup groups (Germany).

Notable: the team played as 'Serbia and Montenegro' (country split mid-tournament),
but martj42's dataset retroactively records the matches as just 'Serbia'.
"""
from __future__ import annotations

from .groups_2022 import bracket_2022 as bracket_2006  # noqa: F401

GROUPS_2006: dict[str, list[str]] = {
    "A": ["Germany", "Ecuador", "Poland", "Costa Rica"],
    "B": ["England", "Sweden", "Paraguay", "Trinidad and Tobago"],
    "C": ["Argentina", "Netherlands", "Ivory Coast", "Serbia"],
    "D": ["Portugal", "Mexico", "Angola", "Iran"],
    "E": ["Italy", "Ghana", "Czech Republic", "United States"],
    "F": ["Brazil", "Australia", "Croatia", "Japan"],
    "G": ["Switzerland", "France", "South Korea", "Togo"],
    "H": ["Spain", "Ukraine", "Tunisia", "Saudi Arabia"],
}

ACTUAL_2006 = {
    "Champion": "Italy",
    "Runner-up": "France",
    "Semi-finalists": ["Germany", "Portugal"],
}
