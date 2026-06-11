# FIFA World Cup 2026 — Prediction Pipeline

Forecast the 2026 World Cup winner from 150 years of international match results, World Football Elo ratings, and a multinomial outcome model. Includes a Monte Carlo simulator that respects FIFA's actual 48-team format (12 groups, top-2 + 8-best-thirds → R32), a multi-tournament backtest validation suite, and tooling for comparing against bookmaker odds and alternative ML algorithms.

> **Validation status:** 7-tournament backtest (1998-2022) hits the eventual champion within the predicted top 6 in **5 of 7 years**. Wins include Spain 2010 at rank #1 and Argentina 2022 at rank #2; misses are Brazil 2002 (rank #9) and Italy 2006 (rank #8). Signal is real but noisy — see the Known limitations section.

---

## Quickstart

```bash
git clone <repo-url> fifa-wc-winner && cd fifa-wc-winner
pip install -r requirements.txt
python baseline.py            # group-stage + chalk-bracket + 10k MC sims
python report.py              # full HTML/markdown report with charts
```

That's enough to get a prediction. The richer workflow follows.

---

## What the pipeline does

```
┌───────────────────┐    ┌─────────────┐    ┌──────────────────┐    ┌──────────────────┐
│ 150 yrs of intl   │    │ World       │    │ Multinomial      │    │ 10,000 Monte     │
│ match results     │───>│ Football    │───>│ logit on Elo     │───>│ Carlo runs of    │
│ (martj42/GitHub)  │    │ Elo         │    │ diff → H/D/A     │    │ the 48-team WC   │
└───────────────────┘    └─────────────┘    └──────────────────┘    └──────────────────┘
```

1. **Data**: ~49k men's full internationals from 1872 onwards, refreshed from `martj42/international_results` on GitHub.
2. **Elo**: World Football Elo with margin-of-victory multiplier and tournament-weighted K (60 for WC, 50 for continental + qualifiers, 20 for friendlies). +100 home boost when not neutral.
3. **Outcome model**: multinomial logistic regression maps Elo diff → P(home win, draw, away win) for the 2006-onward calibration window.
4. **Simulation**: Monte Carlo of the FIFA 2026 format (12 groups → top 2 + 8 best thirds → R32 with cluster-constrained third slotting → R16 → QF → SF → Final + third-place playoff).

---

## Run targets

| Script | Output | Purpose |
| --- | --- | --- |
| `baseline.py` | stdout tables | Group-stage forecasts, chalk-path bracket, 10k MC ranking. |
| `backtest_2022.py` | stdout | Honest 2022 backtest. Argentina rank #2 = validation pass. |
| `backtest_all.py` | stdout | 2014 + 2018 + 2022 freeze-and-predict; multi-year verdict. |
| `compare_market.py` | stdout | Model P(champion) vs bookmaker outright odds, devigged. |
| `compare_models.py` | stdout | Logistic vs GBM vs RF vs MLP vs NB on H/D/A; log_loss + Brier. |
| `report.py` | `report.{md,html}` + `figures/` | Full prediction report with 6 charts. |
| `report_backtests.py` | `report_backtests.{md,html}` | Multi-tournament validation report. |
| `report_market.py` | `report_market.{md,html}` | Market calibration report. |
| `report_models.py` | `report_models.{md,html}` | ML model comparison report. |
| `notebooks/baseline.ipynb` | interactive | Same pipeline in a notebook. |
| `scripts/list_teams.py` | stdout | Verify team-name spellings in the dataset. |

---

## Project structure

```
fifa-wc-winner/
├── README.md                  this file
├── requirements.txt
├── baseline.py                end-to-end CLI runner
├── backtest_2022.py           single-year backtest
├── backtest_all.py            multi-year backtest
├── compare_market.py          model vs bookmaker odds
├── compare_models.py          ML algo comparison
├── report.py                  full prediction report (md + html)
├── report_backtests.py        backtest report (md + html)
├── report_market.py           market calibration report (md + html)
├── report_models.py           ML comparison report (md + html)
├── data/                      cached match data
├── figures/                   generated chart PNGs
├── notebooks/baseline.ipynb   notebook walkthrough
├── scripts/list_teams.py
└── src/
    ├── data.py                results.csv fetcher
    ├── elo.py                 World Football Elo engine
    ├── model.py               multinomial logit outcome model
    ├── simulate.py            Monte Carlo simulator
    ├── predictions.py         match-level + full-bracket prediction tables
    ├── plots.py               chart helpers (bar, funnel, heatmap, bracket, etc.)
    ├── market.py              odds devigging + comparison
    ├── ml_models.py           model zoo for comparison
    ├── report_utils.py        shared HTML/CSS template
    ├── groups_2026.py         official 2026 draw + FIFA R32 bracket
    ├── groups_2022.py         actual 2022 groups + standard 32-team bracket
    ├── groups_2018.py         actual 2018 groups
    └── groups_2014.py         actual 2014 groups
```

---

## Methodology details

### Elo

Standard World Football Elo (eloratings.net convention):

- Starting rating: 1500
- Home advantage: +100 (skipped on neutral grounds)
- Margin-of-victory multiplier: `1` for `|gd| ≤ 1`, `1.5` for `|gd| = 2`, `(11 + |gd|) / 8` otherwise
- K-factor: 60 (FIFA WC), 50 (qualifier / continental), 40 (Nations League / Confederations Cup), 20 (friendly), 30 (default)

### Outcome model

Multinomial logistic regression on `[elo_diff, |elo_diff|, is_friendly, is_wc, is_qualifier]` for matches since 2006. Returns calibrated P(home win, draw, away win). For knockouts, draws are split 50/50 (penalty shootout proxy).

### 2026 Monte Carlo

- 12 groups of 4 (official draw from December 2025).
- Top 2 from each group + 8 best third-placed teams → R32.
- FIFA's third-slot cluster constraints (5 allowed groups per third-slot R32 match) enforced via backtracking matcher.
- R32 ordering encoded from the official spec so adjacent pairs feed the same R16 match.
- Host nations (USA, Canada, Mexico) get +100 Elo for their three group games; knockouts treated as neutral.

---

## Validation

| Year | Champion | Predicted rank | P(champion) | Hit (top 6)? |
| --- | --- | --- | --- | --- |
| 1998 | France | #4 | 8.1% | yes |
| 2002 | Brazil | #9 | 2.8% | **no** |
| 2006 | Italy | #8 | 5.5% | **no** |
| 2010 | Spain | #1 | 22.2% | yes |
| 2014 | Germany | #3 | 15.6% | yes |
| 2018 | France | #6 | 4.7% | yes |
| 2022 | Argentina | #2 | 21.3% | yes |

**Overall: 5/7 top-6 hits.** The two misses (Brazil 2002, Italy 2006) share a pattern — strong squads with depressed pre-tournament Elo from qualification stumbles or recent form dips. Suspected fixes: squad-value Elo augmentation (`experiment_squad_value.py`) and the CONMEBOL K-factor override (`experiment_conmebol_k.py`). Re-run `python backtest_all.py` after any model change.

---

## Known limitations

1. **CONMEBOL Elo inflation.** The CONMEBOL qualification format produces high Elo for mid-tier sides (Ecuador, Colombia, Uruguay). Their 2026 P(champion) > 4% is a model artifact more than playing-strength signal.
2. **Knockout host advantage not modelled.** USA/Mexico/Canada get a group-stage boost but knockouts are treated as neutral, despite most KO matches being on US soil. Probably undercounts host P(SF).
3. **R32 third-slot variance.** When the same 8 thirds qualify across sims, FIFA's cluster constraints often allow multiple valid slottings; the backtracker picks the first valid one. Averages out over 10k sims for aggregate P(champion) but can bias path-dependent stats.
4. **No injury/availability inputs.** A key absence (e.g. starting GK out for the tournament) can swing P(champion) by ~2-3pp and is not in the dataset.
5. **Single-tournament backtest only by default.** 2022 alone may be lucky; run the multi-year backtest before trusting the 2026 numbers.

---

## Comparative study scripts

The repo includes three orthogonal "is the model any good?" checks:

- **vs market** (`compare_market.py` + `report_market.py`): does the model agree with the wisdom of crowds? Large gaps are alpha or bug.
- **vs alternative ML algorithms** (`compare_models.py` + `report_models.py`): does logistic regression underperform gradient boosting? If yes, non-linear interactions are present.
- **vs prior tournaments** (`backtest_all.py` + `report_backtests.py`): is the 2022 result lucky or repeatable?

Run all three before making any large claim about the prediction.

---

## Datasets

Primary source: [martj42/international_results](https://github.com/martj42/international_results) — every men's full international since 1872, refreshed regularly.

Optional Kaggle datasets for richer features (not currently used, but documented as next-step ideas):

- [2026 FIFA World Cup Historical Elo Ratings](https://www.kaggle.com/datasets/afonsofernandescruz/2026-fifa-world-cup-historical-elo-ratings)
- [FIFA World Cup Team Dataset](https://www.kaggle.com/datasets/harrachimustapha/fifa-world-cup-team-dataset)
- [WC2026 Match Probability Baseline Dataset](https://www.kaggle.com/datasets/sarazahran1/wc2026-match-probability-baseline-dataset)

External references:

- [2026 FIFA World Cup — Wikipedia](https://en.wikipedia.org/wiki/2026_FIFA_World_Cup)
- [2026 FIFA World Cup knockout stage — Wikipedia](https://en.wikipedia.org/wiki/2026_FIFA_World_Cup_knockout_stage) (R32 bracket structure)
- [Opta supercomputer predictions](https://theanalyst.com/articles/who-will-win-2026-fifa-world-cup-predictions-opta-supercomputer)

---

## License

This is a personal forecasting project. The match data is sourced from a public CC0 dataset; the model code is yours to fork and adapt.
