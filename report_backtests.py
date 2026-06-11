"""Generate a multi-tournament backtest report (2014, 2018, 2022).

Writes:
    figures/backtests/*.png   per-year top-5 panels
    report_backtests.md
    report_backtests.html

Run:
    python report_backtests.py
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from backtest_all import CONFIGS, run_backtest
from src.data import load_results
from src.report_utils import html_page, img_tag, render_chart

ROOT = Path(__file__).resolve().parent
FIGURES = ROOT / "figures" / "backtests"


def plot_year_panels(summary_rows: list[dict]):
    n = len(summary_rows)
    fig, axes = plt.subplots(1, n, figsize=(5.5 * n, 4.5))
    if n == 1:
        axes = [axes]
    for ax, r in zip(axes, summary_rows):
        teams = [t for t, _ in r["top5"]]
        probs = [p for _, p in r["top5"]]
        teams_r, probs_r = teams[::-1], probs[::-1]
        colors = ["#1a5e1a" if t == r["actual_champion"] else "steelblue"
                  for t in teams_r]
        ax.barh(teams_r, probs_r, color=colors)
        ax.set_xlabel("P(champion)")
        ax.set_title(f"{r['year']}  -  actual: {r['actual_champion']} "
                     f"(rank #{r['predicted_rank']})")
        for i, p in enumerate(probs_r):
            ax.text(p + 0.003, i, f" {p:.1%}", va="center", fontsize=9)
        ax.set_xlim(0, max(probs) * 1.25)
    plt.tight_layout()
    return axes[0]


def main() -> None:
    print("[1/3] Loading match history (slow)...")
    all_results = load_results()

    print("[2/3] Running backtests (one Monte Carlo per year)...")
    summary_rows = []
    for cfg in CONFIGS:
        print(f"  === {cfg.year} ===")
        summary_rows.append(run_backtest(cfg, all_results))

    print("[3/3] Charts + report...")
    _, b64_panels = render_chart(
        lambda: plot_year_panels(summary_rows), FIGURES, "year_panels")

    hits = sum(1 for r in summary_rows if r["predicted_rank"] and r["predicted_rank"] <= 6)
    n = len(summary_rows)
    if hits == n:
        verdict = "Methodology is reliable across tournaments. Predicted champion in top 6 in every year tested."
        verdict_cls = "verdict-good"
    elif hits >= n - 1:
        verdict = (f"Signal is real but noisy. Predicted champion in top 6 in {hits}/{n} years. "
                   "Investigate the miss before trusting 2026 numbers as decision-grade.")
        verdict_cls = "verdict-good"
    else:
        verdict = (f"Insufficient signal. Predicted champion in top 6 in only {hits}/{n} years. "
                   "Methodology needs work; consider adding features or recalibrating Elo K-factors.")
        verdict_cls = "verdict-bad"

    table_rows = []
    for r in summary_rows:
        table_rows.append({
            "Year": r["year"],
            "Actual champion": r["actual_champion"],
            "Predicted rank": r["predicted_rank"],
            "P(champion)": f"{r['predicted_p_champion']:.1%}",
            "P(semi)": f"{r['predicted_p_semi']:.1%}",
        })
    tdf = pd.DataFrame(table_rows)

    md = f"""# Multi-tournament backtest

*Generated: {datetime.now():%Y-%m-%d %H:%M}*
*Tournaments: {', '.join(str(r['year']) for r in summary_rows)}*

## Procedure

For each tournament, run the freeze-and-predict pipeline:

1. Filter the match history to before the tournament start.
2. Recompute Elo from scratch on that subset.
3. Refit the outcome model (multinomial logit on Elo diff features).
4. Run 10,000 Monte Carlo simulations using the actual groups + FIFA bracket.
5. Record where the actual champion ranks in the predicted P(champion) table.

## Summary

{tdf.to_markdown(index=False)}

![Per-year top-5 panels](figures/backtests/year_panels.png)

## Verdict

{verdict}

## Reading the chart

Each panel shows the top-5 predicted teams for that tournament. The actual champion is highlighted in green. If green appears consistently within the top 5, the methodology has real signal across tournaments. If green only appears in one year, that year was likely lucky and the methodology is over-fitted to it.

## Known limitations

- **Team-name spelling drift.** 2014 and 2018 may use legacy spellings (IR Iran, Korea Republic, Cabo Verde). Verify via `python scripts/list_teams.py`.
- **Pre-tournament cutoff is approximate.** Backtests filter on tournament start date; strictly correct would freeze at squad-announcement date.
- **Surprise semi-final runs are not predicted.** Croatia 2018, Morocco 2022 are not expected to surface. Methodology is sound if it ranks the eventual *champion*, not necessarily the SF cast.
"""
    (ROOT / "report_backtests.md").write_text(md, encoding="utf-8")

    table_html = tdf.to_html(index=False, border=0, classes="data-table")
    body = f"""
<h1>Multi-tournament backtest</h1>
<div class="meta">
  Generated: {datetime.now():%Y-%m-%d %H:%M}<br>
  Tournaments: {', '.join(str(r['year']) for r in summary_rows)}
</div>

<div class="exec">
<h3 style="margin-top:0;color:#0c4f0c;">Procedure</h3>
<p>For each tournament, run the freeze-and-predict pipeline: filter history to pre-tournament, recompute Elo, refit the outcome model, run 10,000 Monte Carlo simulations using the actual groups + FIFA bracket, and record where the eventual champion ranks in P(champion).</p>
</div>

<h2>Summary</h2>
{table_html}

{img_tag(b64_panels, "Per-year top-5 panels")}

<h2>Verdict</h2>
<div class="{verdict_cls}">{verdict}</div>

<h2>Reading the chart</h2>
<p>Each panel shows the top-5 predicted teams for that tournament. The actual champion is highlighted in <strong>green</strong>. If green appears consistently within the top 5, the methodology has real signal across tournaments. If green only appears in one year, that year was likely lucky and the methodology is over-fitted to it.</p>

<h2>Known limitations</h2>
<ul>
<li><strong>Team-name spelling drift.</strong> Older tournaments may use legacy spellings (IR Iran, Korea Republic, Cabo Verde). Verify via <code>python scripts/list_teams.py</code>.</li>
<li><strong>Pre-tournament cutoff is approximate.</strong> Strictly correct would freeze at squad-announcement date.</li>
<li><strong>Surprise SF runs are not predicted.</strong> Methodology is sound if it ranks the eventual <em>champion</em>, not necessarily the SF cast.</li>
</ul>
"""
    (ROOT / "report_backtests.html").write_text(
        html_page("Multi-tournament backtest", body), encoding="utf-8")

    print(f"\nDone.\n  {ROOT / 'report_backtests.md'}\n  {ROOT / 'report_backtests.html'}\n  {FIGURES}/")


if __name__ == "__main__":
    main()
