"""Actual 2002 FIFA World Cup groups (South Korea / Japan)."""
from __future__ import annotations

from .groups_2022 import bracket_2022 as bracket_2002  # noqa: F401

GROUPS_2002: dict[str, list[str]] = {
    "A": ["Denmark", "Senegal", "Uruguay", "France"],
    "B": ["Spain", "Paraguay", "South Africa", "Slovenia"],
    "C": ["Brazil", "Turkey", "Costa Rica", "China"],
    "D": ["South Korea", "United States", "Portugal", "Poland"],
    "E": ["Germany", "Republic of Ireland", "Cameroon", "Saudi Arabia"],
    "F": ["Sweden", "England", "Argentina", "Nigeria"],
    "G": ["Mexico", "Italy", "Croatia", "Ecuador"],
    "H": ["Japan", "Belgium", "Russia", "Tunisia"],
}

ACTUAL_2002 = {
    "Champion": "Brazil",
    "Runner-up": "Germany",
    "Semi-finalists": ["Turkey", "South Korea"],
}
