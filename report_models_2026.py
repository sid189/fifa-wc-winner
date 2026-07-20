"""Score every model in the zoo against the completed 2026 World Cup.

report_models.py compares algorithms on a 2021-2022 held-out slice of
historical matches. This script runs the same model zoo (src/ml_models.py)
but scores it against the real 104 matches of the 2026 tournament -- match
results that did not exist anywhere in the training data (data/results.csv
still carries them as NA-score rows, dropped by load_results()). That makes
this a true out-of-sample test, not a historical replay.

Actual scores are pulled from backtest_2026_progress.py (GROUP_STAGE,
KNOCKOUTS), which sourced them from FIFA/ESPN/Al Jazeera on 2026-07-20.

Writes:
    figures/models_2026/*.png
    report_models_2026.md
    report_models_2026.html

Run:
    python report_models_2026.py
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from backtest_2026_progress import GROUP_STAGE, KNOCKOUTS, outcome_label
from src.data import load_results
from src.elo import compute_elo_history
from src.groups_2026 import HOST_TEAMS_2026
from src.ml_models import FEATURE_COLS, get_model_zoo, make_xy
from src.report_utils import html_page, img_tag, render_chart
from src.simulate import HOST_BOOST

ROOT = Path(__file__).resolve().parent
FIGURES = ROOT / "figures" / "models_2026"
CLASSES = ["H", "D", "A"]


def _multiclass_brier(y_true: np.ndarray, p_pred: np.ndarray) -> float:
    onehot = np.zeros_like(p_pred)
    for i, c in enumerate(CLASSES):
        onehot[:, i] = (y_true == c).astype(int)
    return float(((p_pred - onehot) ** 2).sum(axis=1).mean())


def _proba_ordered(model, X: np.ndarray) -> np.ndarray:
    p = model.predict_proba(X)
    class_order = list(model.classes_)
    col_idx = [class_order.index(c) for c in CLASSES]
    return p[:, col_idx]


def build_2026_feature_rows(ratings: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Feature rows for the 72 group matches and 32 knockout matches, using
    the exact frozen pre-tournament ratings (host boost applied to group
    games only, matching report.py / backtest_2026_progress.py)."""
    gs_rows = []
    for team_a, team_b, sa, sb in GROUP_STAGE:
        elo_a = ratings[team_a] + (HOST_BOOST if team_a in HOST_TEAMS_2026 else 0.0)
        elo_b = ratings[team_b] + (HOST_BOOST if team_b in HOST_TEAMS_2026 else 0.0)
        diff = elo_a - elo_b
        gs_rows.append({
            "team_a": team_a, "team_b": team_b, "score": f"{sa}-{sb}",
            "elo_diff": diff, "abs_elo_diff": abs(diff),
            "is_friendly": 0, "is_wc": 1, "is_qualifier": 0,
            "actual": outcome_label(sa, sb),
        })

    ko_rows = []
    for stage, team_a, team_b, sa, sb, shootout_winner in KNOCKOUTS:
        diff = ratings[team_a] - ratings[team_b]
        if shootout_winner is not None:
            actual_a_advances = 1 if shootout_winner == team_a else 0
        else:
            actual_a_advances = 1 if sa > sb else 0
        ko_rows.append({
            "stage": stage, "team_a": team_a, "team_b": team_b,
            "score": f"{sa}-{sb}" + (f" ({shootout_winner} on pens)" if shootout_winner else ""),
            "elo_diff": diff, "abs_elo_diff": abs(diff),
            "is_friendly": 0, "is_wc": 1, "is_qualifier": 0,
            "actual_a_advances": actual_a_advances,
        })
    return pd.DataFrame(gs_rows), pd.DataFrame(ko_rows)


def train_and_evaluate() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, int]:
    print("[1/4] Loading data and computing Elo (slow)...")
    results = load_results()
    ratings, snapshots = compute_elo_history(results, conmebol_qualifier_k_factor=0.5)
    train = snapshots[snapshots["date"] >= pd.Timestamp("2006-01-01")]
    X_train, y_train = make_xy(train)

    print("[2/4] Building 2026 match feature rows (72 group + 32 knockout)...")
    gs_df, ko_df = build_2026_feature_rows(ratings)
    X_gs = gs_df[FEATURE_COLS].to_numpy()
    X_ko = ko_df[FEATURE_COLS].to_numpy()

    zoo = get_model_zoo()
    print(f"[3/4] Training {len(zoo)} models on {len(train):,} matches "
          f"(2006-01-01 -> {train.date.max().date()}), scoring against 2026...")
    rows = []
    for name, model in zoo.items():
        print(f"  fitting {name}...")
        model.fit(X_train, y_train)

        p_gs = _proba_ordered(model, X_gs)
        gs_pred = np.array(CLASSES)[p_gs.argmax(axis=1)]
        gs_hit = gs_pred == gs_df["actual"].to_numpy()
        gs_brier = _multiclass_brier(gs_df["actual"].to_numpy(), p_gs)

        p_ko = _proba_ordered(model, X_ko)
        p_a_advance = p_ko[:, 0] + 0.5 * p_ko[:, 1]  # H + 0.5*D, shootout proxy
        ko_pred_a = p_a_advance >= 0.5
        ko_actual_a = ko_df["actual_a_advances"].to_numpy().astype(bool)
        ko_hit = ko_pred_a == ko_actual_a
        ko_brier = float(((p_a_advance - ko_actual_a.astype(float)) ** 2).mean())

        rows.append({
            "model": name,
            "gs_accuracy": gs_hit.mean(),
            "gs_brier": gs_brier,
            "ko_accuracy": ko_hit.mean(),
            "ko_brier": ko_brier,
            "overall_accuracy": (gs_hit.sum() + ko_hit.sum()) / (len(gs_hit) + len(ko_hit)),
        })

    df = pd.DataFrame(rows).sort_values("ko_brier").reset_index(drop=True)
    return df, gs_df, ko_df, len(train)


def plot_metric(df: pd.DataFrame, metric: str, lower_better: bool = True):
    fig, ax = plt.subplots(figsize=(8, 4))
    sub = df.sort_values(metric, ascending=lower_better).reset_index(drop=True)
    colors = ["#1a5e1a" if i == 0 else "steelblue" for i in range(len(sub))]
    ax.barh(sub["model"], sub[metric], color=colors)
    ax.set_xlabel(metric)
    direction = "lower is better" if lower_better else "higher is better"
    ax.set_title(f"{metric} per model, vs actual 2026 results ({direction})")
    for i, v in enumerate(sub[metric]):
        ax.text(v, i, f"  {v:.4f}", va="center", fontsize=9)
    plt.tight_layout()
    return ax


def write_outputs(df: pd.DataFrame, n_train: int) -> None:
    print("[4/4] Charts + report...")
    _, b64_gsb = render_chart(lambda: plot_metric(df, "gs_brier"), FIGURES, "gs_brier")
    _, b64_kob = render_chart(lambda: plot_metric(df, "ko_brier"), FIGURES, "ko_brier")
    _, b64_acc = render_chart(
        lambda: plot_metric(df, "overall_accuracy", lower_better=False), FIGURES, "overall_accuracy")

    best = df.iloc[0]
    baseline = df[df["model"] == "logistic"].iloc[0]
    gap = baseline["ko_brier"] - best["ko_brier"]
    if best["model"] == "logistic":
        verdict_text = ("Logistic regression -- the model actually used in the pipeline -- "
                        "also comes out on top against the real 2026 results. No swap justified.")
        verdict_cls = "verdict-good"
    elif abs(gap) < 0.01:
        verdict_text = (f"{best['model']} edges out logistic by only {gap:.4f} knockout Brier "
                        "on 32 matches -- within noise for a sample this small. Not enough evidence "
                        "to swap the production model.")
        verdict_cls = "verdict-good"
    else:
        verdict_text = (f"{best['model']} beats logistic by {gap:.4f} knockout Brier against the "
                        "actual 2026 bracket. Worth another look, but re-check on a second tournament "
                        "before swapping -- 32 knockout matches is a small sample.")
        verdict_cls = "verdict-bad"

    disp = df.rename(columns={
        "model": "model", "gs_accuracy": "group_acc", "gs_brier": "group_brier",
        "ko_accuracy": "ko_acc", "ko_brier": "ko_brier", "overall_accuracy": "overall_acc",
    })

    md = f"""# Model zoo vs actual 2026 World Cup results

*Generated: {datetime.now():%Y-%m-%d %H:%M}*

## Setup

- Train: {n_train:,} matches (2006-01-01 -> tournament start), same Elo/features as the production pipeline.
- Test: the actual 104 played matches of the 2026 World Cup (72 group + 32 knockout) -- **true
  out-of-sample**, since these results are not present anywhere in the training data.
- Features: `{', '.join(FEATURE_COLS)}`
- Knockout probability = P(H) + 0.5*P(D), same penalty-shootout proxy the production model uses.

## Results

{disp.to_markdown(index=False, floatfmt=".4f")}

![group_brier](figures/models_2026/gs_brier.png)
![knockout_brier](figures/models_2026/ko_brier.png)
![overall_accuracy](figures/models_2026/overall_accuracy.png)

## Verdict

{verdict_text}

## Reading this vs report_models.py

`report_models.py` scores algorithms on a 2021-2022 slice of *historical* matches the model
never trained on, but which existed at training time in principle. This report scores against
2026 results that postdate every training example -- a stronger, later-in-time test. Treat
`report_models.py` as the "which algorithm is more principled" check and this report as the
"how did the live pick actually do" check.
"""
    (ROOT / "report_models_2026.md").write_text(md, encoding="utf-8")

    table_html = disp.to_html(index=False, border=0, classes="data-table",
                              float_format=lambda x: f"{x:.4f}")
    body = f"""
<h1>Model zoo vs actual 2026 World Cup results</h1>
<div class="meta">Generated: {datetime.now():%Y-%m-%d %H:%M}</div>

<div class="exec">
<h3 style="margin-top:0;color:#0c4f0c;">Setup</h3>
<ul>
<li>Train: {n_train:,} matches (2006-01-01 -> tournament start), same Elo/features as the production pipeline.</li>
<li>Test: the actual 104 played matches of the 2026 World Cup (72 group + 32 knockout) -- true out-of-sample.</li>
<li>Features: <code>{', '.join(FEATURE_COLS)}</code></li>
<li>Knockout probability = P(H) + 0.5*P(D), same penalty-shootout proxy the production model uses.</li>
</ul>
</div>

<h2>Results</h2>
{table_html}

<h2>Charts</h2>
{img_tag(b64_gsb, "group_brier")}
{img_tag(b64_kob, "knockout_brier")}
{img_tag(b64_acc, "overall_accuracy")}

<h2>Verdict</h2>
<div class="{verdict_cls}">{verdict_text}</div>

<h2>Reading this vs report_models.py</h2>
<p><code>report_models.py</code> scores algorithms on a 2021-2022 slice of historical matches
the model never trained on, but which existed at training time in principle. This report scores
against 2026 results that postdate every training example -- a stronger, later-in-time test.
Treat <code>report_models.py</code> as the "which algorithm is more principled" check and this
report as the "how did the live pick actually do" check.</p>
"""
    (ROOT / "report_models_2026.html").write_text(
        html_page("Model zoo vs actual 2026 results", body), encoding="utf-8")


def main() -> None:
    df, gs_df, ko_df, n_train = train_and_evaluate()
    write_outputs(df, n_train)
    print(f"\nDone.\n  {ROOT / 'report_models_2026.md'}\n  {ROOT / 'report_models_2026.html'}\n  {FIGURES}/")


if __name__ == "__main__":
    main()
