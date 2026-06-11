"""Official 2026 FIFA World Cup draw (Dec 5, 2025).

Team-name spellings follow martj42/international_results conventions. The
names commented with '# verify' should be cross-checked against
`python scripts/list_teams.py | grep -i <name>` before running, because
several teams use different spellings across sources.
"""
from __future__ import annotations

GROUPS_2026: dict[str, list[str]] = {
    "A": ["Mexico", "South Africa", "South Korea", "Czech Republic"],  # verify "Czech Republic"
    "B": ["Canada", "Switzerland", "Qatar", "Bosnia and Herzegovina"],
    "C": ["Brazil", "Morocco", "Haiti", "Scotland"],
    "D": ["United States", "Paraguay", "Australia", "Turkey"],  # verify "Turkey" vs "Türkiye"
    "E": ["Germany", "Curaçao", "Ivory Coast", "Ecuador"],  # verify "Ivory Coast"
    "F": ["Netherlands", "Japan", "Tunisia", "Sweden"],
    "G": ["Belgium", "Egypt", "Iran", "New Zealand"],
    "H": ["Spain", "Cape Verde", "Saudi Arabia", "Uruguay"],  # verify "Cape Verde"
    "I": ["France", "Senegal", "Norway", "Iraq"],
    "J": ["Argentina", "Algeria", "Austria", "Jordan"],
    "K": ["Portugal", "Uzbekistan", "Colombia", "DR Congo"],  # verify "DR Congo"
    "L": ["England", "Croatia", "Ghana", "Panama"],
}

# Co-hosts play all three group games in their own country.
HOST_TEAMS_2026: set[str] = {"United States", "Canada", "Mexico"}


# FIFA's official R32 cluster constraints: each of the 8 third-slot R32 matches
# can only be filled by a third-placed team from one of 5 specific groups.
# Slot keys are FIFA match numbers (73-88). Slots for 3rd-placed teams: 74, 77,
# 79, 80, 81, 82, 85, 87.
_THIRD_SLOT_CLUSTERS: dict[int, set[str]] = {
    74: {"A", "B", "C", "D", "F"},
    77: {"C", "D", "F", "G", "H"},
    79: {"C", "E", "F", "H", "I"},
    80: {"E", "H", "I", "J", "K"},
    81: {"B", "E", "F", "I", "J"},
    82: {"A", "E", "H", "I", "J"},
    85: {"E", "F", "G", "I", "J"},
    87: {"D", "E", "I", "J", "L"},
}
_THIRD_SLOTS = [74, 77, 79, 80, 81, 82, 85, 87]


def _assign_thirds(thirds: list[tuple[str, str]]) -> dict[int, str]:
    """Backtracking match of 8 qualifying thirds to FIFA's 8 cluster-bound slots."""
    slot_clusters = [_THIRD_SLOT_CLUSTERS[s] for s in _THIRD_SLOTS]
    n = len(thirds)
    assignment: list[str | None] = [None] * n
    used = [False] * n

    def backtrack(slot_idx: int) -> bool:
        if slot_idx == n:
            return True
        cluster = slot_clusters[slot_idx]
        for i, (team, grp) in enumerate(thirds):
            if not used[i] and grp in cluster:
                assignment[slot_idx] = team
                used[i] = True
                if backtrack(slot_idx + 1):
                    return True
                used[i] = False
        return False

    if not backtrack(0):
        # Cluster constraints unsatisfiable for this draw - fall back to slot order.
        # In practice FIFA's clusters are designed so a valid matching always exists.
        assignment = [t[0] for t in thirds]
    return dict(zip(_THIRD_SLOTS, assignment))


def bracket_2026(
    firsts: dict[str, str],
    seconds: dict[str, str],
    thirds: list[tuple[str, str]],
) -> list[str]:
    """Build the 32-team R32 bracket per FIFA's published structure.

    R32 match numbers (73-88) and the cross-group pairings come from the
    official knockout-stage spec. Adjacent pairs in the returned list feed
    the same R16 match; adjacent R16 winners feed the same QF; etc.
    """
    s = _assign_thirds(thirds)

    matches: dict[int, tuple[str, str]] = {
        73: (seconds["A"], seconds["B"]),
        74: (firsts["E"], s[74]),
        75: (firsts["F"], seconds["C"]),
        76: (firsts["C"], seconds["F"]),
        77: (firsts["I"], s[77]),
        78: (seconds["E"], seconds["I"]),
        79: (firsts["A"], s[79]),
        80: (firsts["L"], s[80]),
        81: (firsts["D"], s[81]),
        82: (firsts["G"], s[82]),
        83: (seconds["K"], seconds["L"]),
        84: (firsts["H"], seconds["J"]),
        85: (firsts["B"], s[85]),
        86: (firsts["J"], seconds["H"]),
        87: (firsts["K"], s[87]),
        88: (seconds["D"], seconds["G"]),
    }

    # R16 pairings (from FIFA spec): M89=W74-W77, M90=W73-W75, M91=W76-W78,
    # M92=W79-W80, M93=W83-W84, M94=W81-W82, M95=W86-W88, M96=W85-W87.
    # Order the 16 R32 matches so adjacent pairs feed the same R16 game.
    r32_order = [74, 77, 73, 75, 76, 78, 79, 80,
                 83, 84, 81, 82, 86, 88, 85, 87]
    bracket: list[str] = []
    for m in r32_order:
        a, b = matches[m]
        bracket.extend([a, b])
    return bracket

