# Music Recommender (Tune Duel)

End-to-end pipeline for modeling 5★ hit probability, user variability, recommendation, and evaluation.

## What’s inside
- **Part 1 (src/part1.py):** Estimate `P(5★ | feature)` (artist/year/popularity + audio/mood features) with **Laplace smoothing**; build per-track `p5_lookup.csv` using empirical + feature aggregation with shrinkage.
- **Part 2 (src/part2.py):** Model time-to-first-5★ `T_u` with **geometric** and **beta-geometric** fits; compare user groups with **Mann–Whitney U** + effect sizes; saves plots.
- **Part 3 (src/part3.py):** Two recommenders:
  - Conditional filtering (taste profile + artist/genre matching)
  - Utility-based sampling (probabilistic, exploration controlled by `T_u`)
- **Part 4 (src/part4.py):** **Monte Carlo** benchmark using `p5_lookup.csv` (Hit@k, AvgRating, `T_u`) + confidence intervals.

## Setup
```bash
pip install pandas numpy scipy matplotlib
