"""Generate a market-calibration report (model P(champion) vs bookmaker odds).

Writes:
    figures/market/*.png   scatter + diff bar
    report_market.md
    report_market.html

REPLACE `CURRENT_ODDS` with current decimal odds before running. Sources:
- Pinnacle (low margin, ~1.04 overround)
- Betfair Exchange (true odds after commission)
- oddsportal.com (aggregator)

Run:
    python report_market.py
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.data import load_results
from src.elo import compute_elo_history
from src.groups_2026 import GROUPS_2026, HOST_TEAMS_2026, bracket_2026
from src.market import compare_to_market, implied_from_decimal, plot_calibration
from src.model import fit_outcome_model
from src.report_utils import html_page, img_tag, render_chart
from src.simulate import monte_carlo

ROOT = Path(__file__).resolve().parent
FIGURES = ROOT / "figures" / "market"


# Decimal outright odds, sourced from ESPN / FanDuel (June 2026).
# Converted from American (Spain +450 -> 5.50) and fractional (Germany 14-1 -> 15.00).
# Italy not included (did not qualify for 2026).
CURRENT_ODDS: dict[str, float] = {
    "Spain":          5.50,
    "France":         5.75,
    "England":        8.00,
    "Portugal":       9.50,
    "Argentina":     10.00,
    "Brazil":        10.50,
    "Germany":       15.00,
    "Netherlands":   21.00,
    "Norway":        36.00,
    "Belgium":       41.00,
    "Colombia":      41.00,
    "Morocco":       51.00,
    "United States": 61.00,
    "Switzerland":   66.00,
    "Uruguay":       66.00,
    "Japan":         66.00,
    "Mexico":        81.00,
    "Ecuador":       81.00,
    "Turkey":        91.00,
    "Croatia":       91.00,
    "Senegal":       91.00,
    "Sweden":       121.00,
    "Austria":      151.00,
    "Canada":       201.00,
}


def plot_diff_bar(cmp_df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(9, 7))
    df = cmp_df.sort_values("diff").reset_index(drop=True)
    colors = ["#cc4444" if d < 0 else "#1a5e1a" for d in df["diff"]]
    ax.barh(df["team"], df["diff"] * 100, color=colors)
    ax.axvline(0, color="0.3", linewidth=0.8)
    ax.set_xlabel("Model - Market (percentage points)")
    ax.set_title("Where model and market disagree (positive = model thinks team is undervalued)")
    for i, d in enumerate(df["diff"]):
        offset = 0.3 if d >= 0 else -0.3
        ax.text(d * 100 + offset, i, f"{d * 100:+.1f}pp",
                va="center", ha="left" if d >= 0 else "right", fontsize=9)
    plt.tight_layout()
    return ax


def _format_cmp(cmp: pd.DataFrame) -> pd.DataFrame:
    out = cmp.copy()
    out["model_p"] = out["model_p"].map(lambda x: f"{x:.1%}")
    out["market_p"] = out["market_p"].map(lambda x: f"{x:.1%}")
    out["diff"] = out["diff"].map(lambda x: f"{x:+.1%}")
    out["ratio"] = out["ratio"].map(lambda x: f"{x:.2f}x")
    return out


def main() -> None:
    print("[1/3] Running 2026 prediction pipeline (slow)...")
    results = load_results()
    ratings, snapshots = compute_elo_history(results)
    model = fit_outcome_model(snapshots, min_date="2006-01-01")
    preds = monte_carlo(
        GROUPS_2026, ratings, model,
        n_sims=10_000, seed=42,
        host_teams=HOST_TEAMS_2026,
        bracket_fn=bracket_2026,
    )

    print("[2/3] Devigging and comparing...")
    market_probs = implied_from_decimal(CURRENT_ODDS)
    overround = sum(1.0 / o for o in CURRENT_ODDS.values())
    cmp = compare_to_market(preds, market_probs, top_n=20).reset_index(drop=True)

    print("[3/3] Charts + report...")
    _, b64_scatter = render_chart(lambda: plot_calibration(cmp), FIGURES, "scatter")
    _, b64_diff = render_chart(lambda: plot_diff_bar(cmp), FIGURES, "diff_bar")

    undervalued = cmp.head(3)
    overvalued = cmp.tail(3).iloc[::-1]
    underv_str = ", ".join(f"{r.team} ({r.diff*100:+.1f}pp)" for r in undervalued.itertuples())
    overv_str = ", ".join(f"{r.team} ({r.diff*100:+.1f}pp)" for r in overvalued.itertuples())

    cmp_md = _format_cmp(cmp)
    md = f"""# Model vs Market - 2026 World Cup

*Generated: {datetime.now():%Y-%m-%d %H:%M}*
*Book overround: {overround:.3f} (1.0 = perfect; >1.05 typical for sportsbooks)*

## Summary

- **Most undervalued by market** (model > market): {underv_str}
- **Most overvalued by market** (model < market): {overv_str}

A positive diff means the model thinks the team is undervalued (potential bet). A negative diff means the market sees something the model does not - usually current form, injuries, or recent friendly results.

## Model vs market

{cmp_md.to_markdown(index=False)}

![Calibration scatter](figures/market/scatter.png)
![Disagreement bar](figures/market/diff_bar.png)

## How to use this

1. **Big positive diffs are either alpha or bugs.** Before betting, ask: does the model see Elo strength the market is missing, or is the model overrating high-Elo teams systematically? Cross-check against `compare_models.py` and `backtest_all.py`.
2. **Big negative diffs are market signals.** If the market prices Brazil 3pp higher than the model, the market knows something. Usually current form, injuries, or recent friendly results that pre-tournament Elo has not yet absorbed.
3. **Devigging method.** This script uses proportional devig (1/odds normalized). For high-margin books switch to Shin's method in `src/market.py`.

## Refresh the odds

`CURRENT_ODDS` at the top of `report_market.py` is a template. Replace with current decimal odds from Pinnacle, Betfair Exchange, or oddsportal.com.
"""
    (ROOT / "report_market.md").write_text(md, encoding="utf-8")

    table_html = cmp_md.to_html(index=False, border=0, classes="data-table")
    body = f"""
<h1>Model vs Market - 2026 World Cup</h1>
<div class="meta">
  Generated: {datetime.now():%Y-%m-%d %H:%M}<br>
  Book overround: {overround:.3f} (1.0 = perfect; &gt;1.05 typical for sportsbooks)
</div>

<div class="exec">
<h3 style="margin-top:0;color:#0c4f0c;">Summary</h3>
<ul>
<li><strong>Most undervalued by market</strong> (model &gt; market): {underv_str}</li>
<li><strong>Most overvalued by market</strong> (model &lt; market): {overv_str}</li>
</ul>
<p>Positive diff = model thinks team is undervalued. Negative diff = market sees something the model does not.</p>
</div>

<h2>Model vs market</h2>
{table_html}

<h2>Charts</h2>
{img_tag(b64_scatter, "Calibration scatter")}
{img_tag(b64_diff, "Disagreement bar")}

<h2>How to use this</h2>
<ol>
<li><strong>Big positive diffs are either alpha or bugs.</strong> Cross-check against <code>compare_models.py</code> and <code>backtest_all.py</code> before sizing a bet.</li>
<li><strong>Big negative diffs are market signals.</strong> The market typically knows about current form, injuries, recent friendly results that pre-tournament Elo has not absorbed.</li>
<li><strong>Devigging method.</strong> Proportional devig used here. For high-margin books switch to Shin's method in <code>src/market.py</code>.</li>
</ol>

<h2>Refresh the odds</h2>
<p>Replace <code>CURRENT_ODDS</code> at the top of <code>report_market.py</code> with current decimal odds from Pinnacle, Betfair Exchange, or oddsportal.com.</p>
"""
    (ROOT / "report_market.html").write_text(
        html_page("Model vs Market - 2026 WC", body), encoding="utf-8")

    print(f"\nDone.\n  {ROOT / 'report_market.md'}\n  {ROOT / 'report_market.html'}\n  {FIGURES}/")


if __name__ == "__main__":
    main()
