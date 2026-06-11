"""Actual 2022 FIFA World Cup groups and R16 bracket.

`bracket_2022` encodes FIFA's standard 32-team cross-group pairing (1A vs 2B,
etc.). The same pattern has been used unchanged since 1986, so 2018 and 2014
backtests import this function directly.
"""
from __future__ import annotations

GROUPS_2022: dict[str, list[str]] = {
    "A": ["Netherlands", "Senegal", "Ecuador", "Qatar"],
    "B": ["England", "United States", "Iran", "Wales"],
    "C": ["Argentina", "Poland", "Mexico", "Saudi Arabia"],
    "D": ["France", "Australia", "Tunisia", "Denmark"],
    "E": ["Japan", "Spain", "Germany", "Costa Rica"],
    "F": ["Morocco", "Croatia", "Belgium", "Canada"],
    "G": ["Brazil", "Switzerland", "Cameroon", "Serbia"],
    "H": ["Portugal", "South Korea", "Uruguay", "Ghana"],
}


def bracket_2022(firsts: dict[str, str], seconds: dict[str, str], thirds: list[tuple[str, str]]) -> list[str]:
    """FIFA standard cross-group R16 pairing.

    Adjacent pairs feed the same QF; adjacent QFs feed the same SF.
    Upper half of the bracket = first 8 slots; lower half = last 8.
    """
    return [
        firsts["A"], seconds["B"],
        firsts["C"], seconds["D"],
        firsts["E"], seconds["F"],
        firsts["G"], seconds["H"],
        firsts["B"], seconds["A"],
        firsts["D"], seconds["C"],
        firsts["F"], seconds["E"],
        firsts["H"], seconds["G"],
    ]
