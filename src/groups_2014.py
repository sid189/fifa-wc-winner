"""Actual 2014 FIFA World Cup groups (Brazil)."""
from __future__ import annotations

from .groups_2022 import bracket_2022 as bracket_2014  # noqa: F401

GROUPS_2014: dict[str, list[str]] = {
    "A": ["Brazil", "Croatia", "Mexico", "Cameroon"],
    "B": ["Spain", "Netherlands", "Chile", "Australia"],
    "C": ["Colombia", "Greece", "Ivory Coast", "Japan"],
    "D": ["Uruguay", "Costa Rica", "England", "Italy"],
    "E": ["Switzerland", "Ecuador", "France", "Honduras"],
    "F": ["Argentina", "Bosnia and Herzegovina", "Iran", "Nigeria"],
    "G": ["Germany", "Portugal", "Ghana", "United States"],
    "H": ["Belgium", "Algeria", "Russia", "South Korea"],
}

ACTUAL_2014 = {
    "Champion": "Germany",
    "Runner-up": "Argentina",
    "Semi-finalists": ["Netherlands", "Brazil"],
}
