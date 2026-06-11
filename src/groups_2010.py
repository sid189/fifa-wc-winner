"""Actual 2010 FIFA World Cup groups (South Africa)."""
from __future__ import annotations

from .groups_2022 import bracket_2022 as bracket_2010  # noqa: F401

GROUPS_2010: dict[str, list[str]] = {
    "A": ["Uruguay", "Mexico", "South Africa", "France"],
    "B": ["Argentina", "South Korea", "Greece", "Nigeria"],
    "C": ["United States", "England", "Slovenia", "Algeria"],
    "D": ["Germany", "Ghana", "Australia", "Serbia"],
    "E": ["Netherlands", "Japan", "Denmark", "Cameroon"],
    "F": ["Paraguay", "Slovakia", "New Zealand", "Italy"],
    "G": ["Brazil", "Portugal", "Ivory Coast", "North Korea"],
    "H": ["Spain", "Chile", "Switzerland", "Honduras"],
}

ACTUAL_2010 = {
    "Champion": "Spain",
    "Runner-up": "Netherlands",
    "Semi-finalists": ["Germany", "Uruguay"],
}
