"""Actual 1998 FIFA World Cup groups (France)."""
from __future__ import annotations

from .groups_2022 import bracket_2022 as bracket_1998  # noqa: F401

GROUPS_1998: dict[str, list[str]] = {
    "A": ["Brazil", "Norway", "Scotland", "Morocco"],
    "B": ["Italy", "Chile", "Cameroon", "Austria"],
    "C": ["France", "Denmark", "South Africa", "Saudi Arabia"],
    "D": ["Spain", "Nigeria", "Paraguay", "Bulgaria"],
    "E": ["Netherlands", "Mexico", "Belgium", "South Korea"],
    "F": ["Germany", "Yugoslavia", "Iran", "United States"],
    "G": ["Romania", "England", "Colombia", "Tunisia"],
    "H": ["Argentina", "Croatia", "Jamaica", "Japan"],
}

ACTUAL_1998 = {
    "Champion": "France",
    "Runner-up": "Brazil",
    "Semi-finalists": ["Croatia", "Netherlands"],
}
