"""Generate a report comparing ML algorithms on the H/D/A outcome task.

Writes:
    figures/models/*.png   metric bar charts
    report_models.md
    report_models.html

Run:
    python report_models.py
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss

from src.data import load_results
from src.elo import compute_elo_history
from src.ml_models import FEATURE_COLS, get_model_zoo, make_xy
from src.report_utils import html_page, img_tag, render_chart

ROOT = Path(__file__).resolve().parent
FIGURES = ROOT / "figures" / "models"
CLASSES = ["H", "D", "A"]


def _multiclass_brier(y_true: np.ndarray, p_pred: np.ndarray) -> float:
    onehot = np.zeros_like(p_pred)
    for i, c in enumerate(CLASSES):
        onehot[:, i] = (y_true == c).astype(int)
    return float(((p_pred - onehot) ** 2).sum(axis=1).mean())


def train_and_evaluate() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    print("[1/3] Loading data and computing Elo (slow)...")
    results = load_results()
    pre = results[results["date"] < pd.Timestamp("2022-11-20")].reset_index(drop=True)
    _, snapshots = compute_elo_history(pre)
    snapshots = snapshots[snapshots["date"] >= pd.Timestamp("2006-01-01")]

    train = snapshots[snapshots["date"] < pd.Timestamp("2021-06-01")]
    test = snapshots[snapshots["date"] >= pd.Timestamp("2021-06-01")]

    X_train, y_train = make_xy(train)
    X_test, y_test = make_xy(test)

    print(f"[2/3] Training {len(get_model_zoo())} models...")
    rows = []
    for name, model in get_model_zoo().items():
        print(f"  fitting {name}...")
        model.fit(X_train, y_train)
        p = model.predict_proba(X_test)
        class_order = list(model.classes_)
        col_idx = [class_order.index(c) for c in CLASSES]
        p_ord = p[:, col_idx]
        rows.append({
            "model": name,
            "log_loss": log_loss(y_test, p_ord, labels=CLASSES),
            "brier": _multiclass_brier(y_test, p_ord),
            "accuracy": accuracy_score(y_test, model.predict(X_test)),
        })

    df = pd.DataFrame(rows).sort_values("log_loss").reset_index(drop=True)
    return df, train, test


def plot_metric(df: pd.DataFrame, metric: str, lower_better: bool = True):
    fig, ax = plt.subplots(figsize=(8, 4))
    sub = df.sort_values(metric, ascending=lower_better).reset_index(drop=True)
    colors = ["#1a5e1a" if i == 0 else "steelblue" for i in range(len(sub))]
    ax.barh(sub["model"], sub[metric], color=colors)
    ax.set_xlabel(metric)
    direction = "lower is better" if lower_better else "higher is better"
    ax.set_title(f"{metric} per model ({direction})")
    for i, v in enumerate(sub[metric]):
        ax.text(v, i, f"  {v:.4f}", va="center", fontsize=9)
    plt.tight_layout()
    return ax


def write_outputs(df: pd.DataFrame, train: pd.DataFrame, test: pd.DataFrame) -> None:
    print("[3/3] Charts + report...")
    _, b64_ll = render_chart(lambda: plot_metric(df, "log_loss"), FIGURES, "log_loss")
    _, b64_br = render_chart(lambda: plot_metric(df, "brier"), FIGURES, "brier")
    _, b64_ac = render_chart(
        lambda: plot_metric(df, "accuracy", lower_better=False), FIGURES, "accuracy")

    best = df.iloc[0]
    baseline = df[df["model"] == "logistic"].iloc[0]
    gap = baseline["log_loss"] - best["log_loss"]
    if best["model"] == "logistic":
        verdict_text = ("Logistic regression remains the best. Feature engineering, "
                        "not algorithm choice, is the lever.")
        verdict_cls = "verdict-good"
    elif gap < 0.005:
        verdict_text = (f"{best['model']} wins narrowly (delta log_loss = {gap:.4f}). "
                        "You are feature-limited; algorithm swap will not move the needle.")
        verdict_cls = "verdict-good"
    else:
        verdict_text = (f"{best['model']} beats logistic by {gap:.4f} log_loss. "
                        "Worth swapping into the pipeline.")
        verdict_cls = "verdict-good"

    md = f"""# ML model comparison

*Generated: {datetime.now():%Y-%m-%d %H:%M}*

## Setup

- Train: {len(train):,} matches ({train.date.min().date()} -> {train.date.max().date()})
- Test:  {len(test):,} matches ({test.date.min().date()} -> {test.date.max().date()})
- Features: `{', '.join(FEATURE_COLS)}`
- Task: multinomial classification {{H, D, A}}

## Results

{df.to_markdown(index=False, floatfmt=".4f")}

![log_loss](figures/models/log_loss.png)
![brier](figures/models/brier.png)
![accuracy](figures/models/accuracy.png)

## Verdict

{verdict_text}

## Reading the gap

- All models tie within 0.005 log_loss -> **feature-limited**; add features (form, market value, xG).
- Gradient boosting wins by >0.01 -> non-linear interactions exist; swap GBM into the pipeline.
- MLP wins -> calibrate via `CalibratedClassifierCV` before swapping (probabilities may not be honest).
"""
    (ROOT / "report_models.md").write_text(md, encoding="utf-8")

    table_html = df.to_html(index=False, border=0, classes="data-table",
                            float_format=lambda x: f"{x:.4f}")
    body = f"""
<h1>ML model comparison</h1>
<div class="meta">Generated: {datetime.now():%Y-%m-%d %H:%M}</div>

<div class="exec">
<h3 style="margin-top:0;color:#0c4f0c;">Setup</h3>
<ul>
<li>Train: {len(train):,} matches ({train.date.min().date()} -> {train.date.max().date()})</li>
<li>Test:  {len(test):,} matches ({test.date.min().date()} -> {test.date.max().date()})</li>
<li>Features: <code>{', '.join(FEATURE_COLS)}</code></li>
<li>Task: multinomial classification {{H, D, A}}</li>
</ul>
</div>

<h2>Results</h2>
{table_html}

<h2>Charts</h2>
{img_tag(b64_ll, "log_loss")}
{img_tag(b64_br, "brier")}
{img_tag(b64_ac, "accuracy")}

<h2>Verdict</h2>
<div class="{verdict_cls}">{verdict_text}</div>

<h2>Reading the gap</h2>
<ul>
<li>All models tie within 0.005 log_loss -> <strong>feature-limited</strong>. Add features (form, market value, xG) rather than tuning algorithms.</li>
<li>Gradient boosting wins by &gt;0.01 -> non-linear interactions exist; swap GBM into the pipeline.</li>
<li>MLP wins -> calibrate via <code>CalibratedClassifierCV</code> before swapping (raw MLP probabilities are often miscalibrated).</li>
</ul>
"""
    (ROOT / "report_models.html").write_text(
        html_page("ML model comparison", body), encoding="utf-8")


def main() -> None:
    df, train, test = train_and_evaluate()
    write_outputs(df, train, test)
    print(f"\nDone.\n  {ROOT / 'report_models.md'}\n  {ROOT / 'report_models.html'}\n  {FIGURES}/")


if __name__ == "__main__":
    main()
