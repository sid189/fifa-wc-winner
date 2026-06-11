"""Shared utilities for report scripts.

Provides a common HTML template, chart-rendering helper, and table formatters
so each report-style script is mostly content rather than HTML scaffolding.
"""
from __future__ import annotations

import base64
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt


CSS = """
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
.caveat-list { list-style: none; padding-left: 0; }
.caveat-list li { background: #fff5f0; border-left: 4px solid #cc6644;
                  padding: 0.6em 1em; margin: 0.5em 0; border-radius: 0 4px 4px 0; }
code { background: #f0f0f0; padding: 1px 5px; border-radius: 3px; font-size: 0.9em; }
.muted { color: #888; }
.section-note { color: #555; font-size: 0.95em; margin: 0.5em 0 1em; }
.verdict-good { background: #f0fff0; border-left: 4px solid #1a5e1a;
                padding: 0.6em 1em; border-radius: 0 4px 4px 0; }
.verdict-bad  { background: #fff5f0; border-left: 4px solid #cc6644;
                padding: 0.6em 1em; border-radius: 0 4px 4px 0; }
"""


def render_chart(make_chart: Callable, figures_dir: Path, name: str) -> tuple[Path, str]:
    """Run a chart function, save PNG, return (path, base64_string)."""
    figures_dir.mkdir(parents=True, exist_ok=True)
    ax = make_chart()
    fig = ax.get_figure() if hasattr(ax, "get_figure") else ax
    path = figures_dir / f"{name}.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    b64 = base64.b64encode(path.read_bytes()).decode("ascii")
    return path, b64


def img_tag(b64: str, alt: str) -> str:
    return f'<img src="data:image/png;base64,{b64}" alt="{alt}">'


def html_page(title: str, body: str) -> str:
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{title}</title>
<style>{CSS}</style>
</head>
<body>
{body}
</body>
</html>
"""
