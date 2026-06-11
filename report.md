# FIFA World Cup 2026 - Prediction Report

*Generated: 2026-06-11 08:36*
*Methodology: World Football Elo + multinomial logit + 10,000 Monte Carlo simulations*
*Backtest validation: 2022 (Argentina ranked #2 - methodology has signal)*

---

## Executive summary

- **Predicted champion**: Spain (21.4%)
- **Top 5**: **Spain** (21.4%), **Argentina** (16.1%), **France** (9.6%), **England** (7.7%), **Brazil** (5.9%)
- **Chalk-path final**: France vs Spain -> Spain
- **Chalk-path third place**: England vs Argentina -> Argentina
- **Chalk-path semifinals**: France vs England | Spain vs Argentina
- **Surprise teams in top 15**: Ecuador (2.6%), Norway (2.4%), Turkey (2.2%), Morocco (2.0%)
- **Bracket asymmetry**: upper half 44% vs lower half 55%. The lower bracket is meaningfully stronger; bottom-half teams face tougher paths.

## Top 15 contenders

![Top 15 P(champion)](figures/top_champions.png)

| team        |      elo |   P(R16) |   P(QF) |   P(SF) |   P(F) |   P(Champ) |
|:------------|---------:|---------:|--------:|--------:|-------:|-----------:|
| Spain       | 2237.789 |    0.761 |   0.556 |   0.448 |  0.310 |      0.214 |
| Argentina   | 2214.728 |    0.695 |   0.546 |   0.405 |  0.244 |      0.161 |
| France      | 2141.706 |    0.695 |   0.447 |   0.295 |  0.185 |      0.096 |
| England     | 2122.337 |    0.675 |   0.413 |   0.250 |  0.144 |      0.077 |
| Brazil      | 2093.792 |    0.614 |   0.384 |   0.221 |  0.125 |      0.059 |
| Colombia    | 2087.762 |    0.611 |   0.355 |   0.191 |  0.095 |      0.051 |
| Germany     | 2017.750 |    0.557 |   0.281 |   0.146 |  0.071 |      0.030 |
| Ecuador     | 2023.848 |    0.535 |   0.275 |   0.144 |  0.069 |      0.026 |
| Mexico      | 1967.146 |    0.613 |   0.309 |   0.143 |  0.067 |      0.026 |
| Portugal    | 2027.190 |    0.522 |   0.264 |   0.130 |  0.055 |      0.025 |
| Netherlands | 2016.783 |    0.476 |   0.267 |   0.128 |  0.062 |      0.025 |
| Norway      | 2023.474 |    0.488 |   0.253 |   0.131 |  0.062 |      0.024 |
| Turkey      | 2010.546 |    0.514 |   0.264 |   0.112 |  0.049 |      0.022 |
| Morocco     | 2004.358 |    0.457 |   0.244 |   0.116 |  0.051 |      0.020 |
| Japan       | 2010.127 |    0.454 |   0.247 |   0.115 |  0.053 |      0.020 |

## Tournament progression funnel

![Probability funnel](figures/funnel.png)

The funnel decomposes each contender's championship probability by stage. A large drop between consecutive bars marks the round where that team typically gets eliminated.

## Chalk-path bracket

![Chalk bracket](figures/bracket_chalk.png)

Deterministic "every favorite advances" view. Useful for spotting bracket structure, but any single chalk path happens <1% of the time across 10k sims.

## Head-to-head matrix (top 8)

![H2H top 8](figures/h2h_top8.png)

Pairwise knockout win probability on neutral ground (draws split 50/50 for penalty shootouts).

## Elo trajectory since 2022

![Elo trajectory](figures/elo_trajectory.png)

Diagnostic view of how the top-8 contenders' ratings have evolved over the last cycle.

## Confederation breakdown

![Confederation P(champion)](figures/confederation.png)

UEFA's dominance is expected; the key question is how concentrated the CONMEBOL slice is in Argentina/Brazil vs spread across Ecuador/Colombia/Uruguay.

## Group stage highlights

**Top 5 matches most likely to end in a draw:**

| group   | team_a     | team_b       |   p_draw |   p_a_win |   p_b_win |
|:--------|:-----------|:-------------|---------:|----------:|----------:|
| D       | Paraguay   | Australia    |     0.29 |      0.35 |      0.36 |
| F       | Tunisia    | Sweden       |     0.29 |      0.36 |      0.35 |
| E       | Germany    | Ecuador      |     0.29 |      0.37 |      0.34 |
| J       | Algeria    | Austria      |     0.29 |      0.37 |      0.34 |
| H       | Cape Verde | Saudi Arabia |     0.29 |      0.32 |      0.39 |

## Risk factors and caveats

1. **Spain at 21.4%** is high relative to typical market pricing. Run `python compare_market.py` after updating with current odds.
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
