"""Generate a full 2026 prediction report in both markdown and HTML.

Writes:
    figures/*.png   six charts
    report.md       markdown report (image refs to figures/)
    report.html     single-file HTML report (charts embedded as base64)

Run:
    python report.py
"""
from __future__ import annotations

import base64
from datetime import datetime
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt

from src.data import load_results
from src.elo import compute_elo_history
from src.model import fit_outcome_model
from src.simulate import monte_carlo
from src.groups_2026 import GROUPS_2026, HOST_TEAMS_2026, bracket_2026
from src.predictions import (
    group_stage_predictions,
    draw_watch,
    most_likely_seeding,
    full_bracket_predictions,
)
from src.plots import (
    plot_champion_bar,
    plot_funnel,
    plot_h2h_heatmap,
    plot_elo_trajectory,
    plot_chalk_bracket,
    plot_confederation_breakdown,
)

ROOT = Path(__file__).resolve().parent
FIGURES = ROOT / "figures"
FIGURES.mkdir(exist_ok=True)


CONFEDERATIONS: dict[str, list[str]] = {
    "UEFA": ["Spain", "France", "England", "Germany", "Portugal", "Netherlands",
             "Belgium", "Croatia", "Switzerland", "Norway", "Sweden", "Denmark",
             "Austria", "Czech Republic", "Scotland", "Bosnia and Herzegovina",
             "Turkey"],
    "CONMEBOL": ["Argentina", "Brazil", "Colombia", "Ecuador", "Uruguay", "Paraguay"],
    "AFC": ["Japan", "South Korea", "Iran", "Saudi Arabia", "Iraq", "Jordan",
            "Qatar", "Uzbekistan", "Australia"],
    "CAF": ["Morocco", "Senegal", "Tunisia", "Egypt", "Algeria", "Cape Verde",
            "Ivory Coast", "South Africa", "DR Congo", "Ghana"],
    "CONCACAF": ["Mexico", "United States", "Canada", "Panama", "Haiti", "Curaçao"],
    "OFC": ["New Zealand"],
}


def _render(make_chart: Callable, name: str) -> tuple[Path, str]:
    """Run a chart fn, save PNG, return (path, base64_string)."""
    ax = make_chart()
    fig = ax.get_figure()
    path = FIGURES / f"{name}.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    b64 = base64.b64encode(path.read_bytes()).decode("ascii")
    return path, b64


def main() -> None:
    print("[1/5] Running data pipeline...")
    results = load_results()
    ratings, snapshots = compute_elo_history(results, conmebol_qualifier_k_factor=0.5)
    model = fit_outcome_model(snapshots, min_date="2006-01-01")

    print("[2/5] Monte Carlo (10k sims)...")
    preds = monte_carlo(
        GROUPS_2026, ratings, model,
        n_sims=10_000, seed=42,
        host_teams=HOST_TEAMS_2026,
        host_ko_boost=50.0,
        bracket_fn=bracket_2026,
    )

    gs = group_stage_predictions(GROUPS_2026, ratings, model,
                                 host_teams=HOST_TEAMS_2026)
    draws = draw_watch(gs, top_n=8)
    firsts, seconds, thirds = most_likely_seeding(
        GROUPS_2026, ratings, host_teams=HOST_TEAMS_2026)
    bracket = bracket_2026(firsts, seconds, thirds)
    full = full_bracket_predictions(bracket, ratings, model)

    print("[3/5] Generating charts...")
    top8 = preds.head(8)["team"].tolist()
    chart_paths: dict[str, Path] = {}
    chart_b64: dict[str, str] = {}
    chart_specs = [
        ("top_champions",   lambda: plot_champion_bar(preds, top_n=15)),
        ("funnel",          lambda: plot_funnel(preds, top_n=10)),
        ("h2h_top8",        lambda: plot_h2h_heatmap(top8, ratings, model)),
        ("elo_trajectory",  lambda: plot_elo_trajectory(snapshots, top8, since="2022-01-01")),
        ("bracket_chalk",   lambda: plot_chalk_bracket(full)),
        ("confederation",   lambda: plot_confederation_breakdown(preds, CONFEDERATIONS)),
    ]
    for name, fn in chart_specs:
        path, b64 = _render(fn, name)
        chart_paths[name] = path
        chart_b64[name] = b64

    print("[4/5] Writing report.md...")
    write_markdown(preds, draws, full, chart_paths)

    print("[5/5] Writing report.html...")
    write_html(preds, draws, full, chart_b64)

    print(f"\nDone.\n  {ROOT / 'report.md'}\n  {ROOT / 'report.html'}\n  {FIGURES}/")


# ---------------------------------------------------------------------------
# Shared analysis helpers (used by both renderers)
# ---------------------------------------------------------------------------

def _format_top5(preds, bold: bool = True) -> str:
    fmt = "**{}** ({:.1%})" if bold else "{} ({:.1%})"
    return ", ".join(fmt.format(r.team, r.p_champion)
                     for r in preds.head(5).itertuples())


def _identify_surprises(preds) -> str:
    historical_minnows = {"Ecuador", "Norway", "Turkey", "Morocco", "Switzerland",
                          "Curaçao", "Cape Verde", "Ivory Coast"}
    surprises = preds.head(15).query("team in @historical_minnows")
    if surprises.empty:
        return "(none in top 15)"
    return ", ".join(f"{r.team} ({r.p_champion:.1%})" for r in surprises.itertuples())


def _bracket_summary(full) -> tuple[str, str, str]:
    f_row = full[full["round"] == "F"].iloc[0]
    third = full[full["round"] == "3rd"]
    chalk_final = f"{f_row['team_a']} vs {f_row['team_b']} -> {f_row['favorite']}"
    if not third.empty:
        t_row = third.iloc[0]
        third_str = f"{t_row['team_a']} vs {t_row['team_b']} -> {t_row['favorite']}"
    else:
        third_str = "n/a"
    sf = full[full["round"] == "SF"]
    sf_str = " | ".join(f"{r['team_a']} vs {r['team_b']}" for _, r in sf.iterrows())
    return chalk_final, third_str, sf_str


def _bracket_asymmetry(preds, full) -> tuple[float, float, str]:
    r32 = full[full["round"] == "R32"].reset_index(drop=True)
    upper, lower = [], []
    for i, r in r32.iterrows():
        (upper if i < 8 else lower).extend([r["team_a"], r["team_b"]])
    up = float(preds[preds["team"].isin(upper)]["p_champion"].sum())
    lo = float(preds[preds["team"].isin(lower)]["p_champion"].sum())
    if up > lo + 0.05:
        verdict = "The upper bracket is meaningfully stronger; top-half teams face tougher paths."
    elif lo > up + 0.05:
        verdict = "The lower bracket is meaningfully stronger; bottom-half teams face tougher paths."
    else:
        verdict = "Roughly symmetric; neither side has a path advantage."
    return up, lo, verdict


def _top15_table(preds):
    return (
        preds.head(15)[["team", "elo", "p_r16", "p_qf", "p_semi", "p_final", "p_champion"]]
        .rename(columns={"p_r16": "P(R16)", "p_qf": "P(QF)",
                         "p_semi": "P(SF)", "p_final": "P(F)",
                         "p_champion": "P(Champ)"})
    )


# ---------------------------------------------------------------------------
# Markdown renderer
# ---------------------------------------------------------------------------

def write_markdown(preds, draws, full, chart_paths) -> None:
    champion = preds.iloc[0]["team"]
    champion_p = preds.iloc[0]["p_champion"]
    chalk_final, third_place, semis = _bracket_summary(full)
    up, lo, verdict = _bracket_asymmetry(preds, full)

    draws_md = draws.head(5).to_markdown(index=False, floatfmt=".2f")
    top15_md = _top15_table(preds).to_markdown(index=False, floatfmt=".3f")

    md = f"""# FIFA World Cup 2026 - Prediction Report

*Generated: {datetime.now():%Y-%m-%d %H:%M}*
*Methodology: World Football Elo + multinomial logit + 10,000 Monte Carlo simulations*
*Backtest validation: 2022 (Argentina ranked #2 - methodology has signal)*

---

## Executive summary

- **Predicted champion**: {champion} ({champion_p:.1%})
- **Top 5**: {_format_top5(preds)}
- **Chalk-path final**: {chalk_final}
- **Chalk-path third place**: {third_place}
- **Chalk-path semifinals**: {semis}
- **Surprise teams in top 15**: {_identify_surprises(preds)}
- **Bracket asymmetry**: upper half {up:.0%} vs lower half {lo:.0%}. {verdict}

## Top 15 contenders

![Top 15 P(champion)](figures/{chart_paths['top_champions'].name})

{top15_md}

## Tournament progression funnel

![Probability funnel](figures/{chart_paths['funnel'].name})

The funnel decomposes each contender's championship probability by stage. A large drop between consecutive bars marks the round where that team typically gets eliminated.

## Chalk-path bracket

![Chalk bracket](figures/{chart_paths['bracket_chalk'].name})

Deterministic "every favorite advances" view. Useful for spotting bracket structure, but any single chalk path happens <1% of the time across 10k sims.

## Head-to-head matrix (top 8)

![H2H top 8](figures/{chart_paths['h2h_top8'].name})

Pairwise knockout win probability on neutral ground (draws split 50/50 for penalty shootouts).

## Elo trajectory since 2022

![Elo trajectory](figures/{chart_paths['elo_trajectory'].name})

Diagnostic view of how the top-8 contenders' ratings have evolved over the last cycle.

## Confederation breakdown

![Confederation P(champion)](figures/{chart_paths['confederation'].name})

UEFA's dominance is expected; the key question is how concentrated the CONMEBOL slice is in Argentina/Brazil vs spread across Ecuador/Colombia/Uruguay.

## Group stage highlights

**Top 5 matches most likely to end in a draw:**

{draws_md}

## Risk factors and caveats

1. **{champion} at {champion_p:.1%}** is high relative to typical market pricing. Run `python compare_market.py` after updating with current odds.
2. **Host advantage only applies in group stage.** Knockouts treated as neutral despite most games being in the USA.
3. **CONMEBOL Elo inflation.** Treat Ecuador/Colombia/Uruguay championship probabilities with extra skepticism.
4. **Single-tournament backtest.** Argentina #2 in 2022 is strong, but n=1. Run `python backtest_all.py`.
5. **Random third-team slotting variance.** Cluster constraints often allow multiple valid slottings; the backtracker picks the first one.

## How to interpret these numbers

- **P(champion)** is the model's outright win probability. Compare to bookmaker odds via `src.market.implied_from_decimal`.
- **P(SF), P(QF), P(R16)** have lower variance than the outright.
- **Chalk path** is narrative-useful, betting-useless. Use the Monte Carlo table for sizing.

## Next steps

1. Re-run `compare_market.py` with current bookmaker odds.
2. Re-run `compare_models.py` to see if GBM or random forest beat logistic regression materially.
3. Add features beyond Elo (rolling form, squad market value, key player availability).
4. Validate across multiple tournaments (`backtest_all.py`) before trusting 2026 numbers.
"""

    (ROOT / "report.md").write_text(md, encoding="utf-8")


# ---------------------------------------------------------------------------
# HTML renderer (single file, base64-embedded charts)
# ---------------------------------------------------------------------------

_CSS = """
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
       max-width: 1100px; margin: 2em auto; padding: 0 1.5em; color: #222; line-height: 1.55;
       background: #fafafa; }
h1 { border-bottom: 2px solid #1a5e1a; padding-bottom: 0.3em; color: #0c4f0c; }
h2 { margin-top: 2.2em; color: #1a5e1a; border-bottom: 1px solid #ddd; padding-bottom: 0.2em; }
h3 { margin-top: 1.4em; color: #444; }
img { max-width: 100%; height: auto; display: block; margin: 1.2em auto;
      border: 1px solid #e0e0e0; background: #fff; padding: 8px; border-radius: 4px; }
table { border-collapse: collapse; width: 100%; margin: 1em 0; font-size: 0.9em;
        background: #fff; box-shadow: 0 1px 3px rgba(0,0,0,0.04); }
th, td { padding: 7px 10px; border-bottom: 1px solid #eee; text-align: right; }
th { background: #f4f4f4; font-weight: 600; text-align: center; border-bottom: 2px solid #ccc; }
td:first-child, th:first-child { text-align: left; }
tr:hover td { background: #fafff5; }
.meta { color: #777; font-style: italic; font-size: 0.92em; margin-bottom: 1em; }
.exec { background: #fff; border-left: 4px solid #1a5e1a; padding: 1em 1.4em;
        border-radius: 0 4px 4px 0; box-shadow: 0 1px 3px rgba(0,0,0,0.04); }
.exec ul { margin: 0.5em 0; padding-left: 1.5em; }
.exec li { margin: 0.3em 0; }
.caveat { background: #fff5f0; border-left: 4px solid #cc6644; padding: 0.4em 1em;
          border-radius: 0 4px 4px 0; margin: 0.5em 0; }
.caveat-list { list-style: none; padding-left: 0; }
.caveat-list li { background: #fff5f0; border-left: 4px solid #cc6644;
                  padding: 0.6em 1em; margin: 0.5em 0; border-radius: 0 4px 4px 0; }
code { background: #f0f0f0; padding: 1px 5px; border-radius: 3px; font-size: 0.9em; }
.muted { color: #888; }
.section-note { color: #555; font-size: 0.95em; margin: 0.5em 0 1em; }
"""


def _img(b64: str, alt: str) -> str:
    return f'<img src="data:image/png;base64,{b64}" alt="{alt}">'


def _table_to_html(df) -> str:
    out = df.copy()
    pct_cols = [c for c in out.columns if c.startswith("P(")]
    for c in pct_cols:
        out[c] = out[c].map(lambda x: f"{x:.1%}")
    if "elo" in out.columns:
        out["elo"] = out["elo"].map(lambda x: f"{x:.0f}")
    return out.to_html(index=False, border=0, classes="data-table")


def _draws_to_html(draws) -> str:
    df = draws.head(5).copy()
    for c in ["p_draw", "p_a_win", "p_b_win"]:
        df[c] = df[c].map(lambda x: f"{x:.2f}")
    df.columns = ["Group", "Team A", "Team B", "P(Draw)", "P(A win)", "P(B win)"]
    return df.to_html(index=False, border=0, classes="data-table")


def write_html(preds, draws, full, chart_b64) -> None:
    champion = preds.iloc[0]["team"]
    champion_p = preds.iloc[0]["p_champion"]
    chalk_final, third_place, semis = _bracket_summary(full)
    up, lo, verdict = _bracket_asymmetry(preds, full)

    top15_html = _table_to_html(_top15_table(preds))
    draws_html = _draws_to_html(draws)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>FIFA World Cup 2026 - Prediction Report</title>
<style>{_CSS}</style>
</head>
<body>

<h1>FIFA World Cup 2026 - Prediction Report</h1>
<div class="meta">
  Generated: {datetime.now():%Y-%m-%d %H:%M}<br>
  Methodology: World Football Elo + multinomial logit + 10,000 Monte Carlo simulations<br>
  Backtest validation: 2022 (Argentina ranked #2 - methodology has signal)
</div>

<div class="exec">
  <h3 style="margin-top:0;color:#0c4f0c;">Executive summary</h3>
  <ul>
    <li><strong>Predicted champion:</strong> {champion} ({champion_p:.1%})</li>
    <li><strong>Top 5:</strong> {_format_top5(preds, bold=False)}</li>
    <li><strong>Chalk-path final:</strong> {chalk_final}</li>
    <li><strong>Chalk-path third place:</strong> {third_place}</li>
    <li><strong>Chalk-path semifinals:</strong> {semis}</li>
    <li><strong>Surprise teams in top 15:</strong> {_identify_surprises(preds)}</li>
    <li><strong>Bracket asymmetry:</strong> upper half {up:.0%} vs lower half {lo:.0%}. <em>{verdict}</em></li>
  </ul>
</div>

<h2>Top 15 contenders</h2>
{_img(chart_b64['top_champions'], 'Top 15 P(champion)')}
{top15_html}

<h2>Tournament progression funnel</h2>
<p class="section-note">The funnel decomposes each contender's championship probability by stage.
A large drop between consecutive bars marks the round where that team typically gets eliminated.
Teams whose P(reach R16) is high but P(reach QF) drops sharply are stuck in a weak group followed
by a tough R16 draw.</p>
{_img(chart_b64['funnel'], 'Probability funnel')}

<h2>Chalk-path bracket</h2>
<p class="section-note">Deterministic "every favorite advances" view. Useful for spotting bracket
structure (where the heavyweights collide), but <strong>not a confident prediction</strong> -
any single chalk path happens &lt;1% of the time across the 10k Monte Carlo runs.</p>
{_img(chart_b64['bracket_chalk'], 'Chalk-path bracket')}

<h2>Head-to-head matrix (top 8)</h2>
<p class="section-note">Pairwise knockout win probability on neutral ground (penalties split 50/50).
Cells near 0.5 = coin flips; strong asymmetric rows = clear favorites.</p>
{_img(chart_b64['h2h_top8'], 'H2H top 8')}

<h2>Elo trajectory since 2022</h2>
<p class="section-note">Diagnostic view of how the top-8 contenders' ratings have evolved over the
last cycle. Sudden cliffs typically come from upset losses at major tournaments (legitimate Elo)
or mis-weighted matches in the dataset (bug). Steady upward trends suggest a team peaking at the
right time.</p>
{_img(chart_b64['elo_trajectory'], 'Elo trajectory')}

<h2>Confederation breakdown</h2>
<p class="section-note">Aggregated championship probability by confederation. UEFA's dominance is
expected; the key question is how concentrated the CONMEBOL slice is in Argentina/Brazil vs spread
across Ecuador/Colombia/Uruguay (the model has historically inflated CONMEBOL Elo).</p>
{_img(chart_b64['confederation'], 'Confederation breakdown')}

<h2>Group stage highlights</h2>
<h3>Top 5 matches most likely to end in a draw</h3>
{draws_html}
<p class="section-note">Draw-heavy matches typically pair two teams of similar Elo and contained
playing styles. Any draw against a group favorite is a 1-point gift to the underdog and a hit to
the favorite's qualification probability.</p>

<h2>Risk factors and caveats</h2>
<ul class="caveat-list">
  <li><strong>{champion} at {champion_p:.1%}</strong> is high relative to typical market pricing.
      Run <code>python compare_market.py</code> after updating with current odds to see the
      model-vs-market gap explicitly.</li>
  <li><strong>Host advantage only applies in group stage.</strong> USA/Canada/Mexico get +100 Elo
      for their three group games, but knockouts are treated as neutral. Since most KO games are
      in the USA, this likely under-weights the US's path.</li>
  <li><strong>CONMEBOL Elo inflation.</strong> The model places Ecuador, Colombia, and Uruguay
      in the top 10-15. Treat any CONMEBOL team outside Argentina/Brazil with extra skepticism.</li>
  <li><strong>Single-tournament backtest.</strong> Argentina #2 in 2022 is strong, but n=1. Run
      <code>python backtest_all.py</code> (2014 + 2018 + 2022) before trusting these numbers as
      decision-grade.</li>
  <li><strong>Random third-team slotting variance.</strong> When the same set of 8 thirds qualifies,
      FIFA's cluster constraints often allow multiple valid slottings. The backtracker picks the
      first one it finds, which may bias individual team paths.</li>
</ul>

<h2>How to interpret these numbers</h2>
<ul>
  <li><strong>P(champion)</strong> is the model's outright win probability. Roughly compare to
      bookmaker decimal odds: <code>1/odds</code> gives implied probability; devig with
      <code>src.market.implied_from_decimal</code>.</li>
  <li><strong>P(SF), P(QF), P(R16)</strong> are lower-variance bets. Markets price these less
      efficiently than the outright.</li>
  <li><strong>Chalk path</strong> is a stylized walk - useful for narrative, useless for sizing.
      Use the Monte Carlo table for actual probabilities.</li>
</ul>

<h2>Next steps to tighten the prediction</h2>
<ol>
  <li>Re-run <code>compare_market.py</code> with current bookmaker odds.</li>
  <li>Re-run <code>compare_models.py</code> to see if GBM or random forest materially beats logistic regression.</li>
  <li>Add features beyond Elo (rolling form, squad market value, key player availability).</li>
  <li>Validate across multiple tournaments via <code>backtest_all.py</code>.</li>
</ol>

</body>
</html>
"""

    (ROOT / "report.html").write_text(html, encoding="utf-8")


if __name__ == "__main__":
    main()
