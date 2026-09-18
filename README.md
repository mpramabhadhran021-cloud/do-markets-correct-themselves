# Do Markets Correct Themselves?

## What is this project?

This project studies how crude oil, wheat, and copper prices respond after large shocks.

The idea is motivated by Adam Smith's Invisible Hand, but the project does not try to prove or disprove that broader idea.

## Research Question

> Do commodity prices tend to move back toward earlier levels after large shocks?

## Data

I use monthly data from FRED and the U.S. Energy Information Administration. The sample period depends on the series and mainly covers the 1980s or 1990s to 2026.

## Method

I use descriptive statistics, plots, correlations, lag analysis, simple regressions, autocorrelation analysis, shock analysis, bootstrap checks, and comparisons across markets.

For each shock, I measure the size of the price movement and how long it takes to move back toward the earlier price level.

## Main Findings

The three markets show evidence of price adjustment after some large shocks, but the speed of adjustment differs across markets.

Some shocks do not return to the earlier price level within the 24-month window used in the analysis.

These results are evidence of price adjustment, not proof that markets always correct themselves.

## Repository

data/ — raw and processed data
scripts/data_prep.py — data preparation
notebooks/01_market_data_analysis.ipynb — first analysis
notebooks/02_market_self_correction_analysis.ipynb — shock and self-correction analysis

## How to Run

python scripts/data_prep.py
jupyter notebook notebooks/01_market_data_analysis.ipynb
jupyter notebook notebooks/02_market_self_correction_analysis.ipynb

## Limitations

This is an observational analysis and cannot establish causality. The definition of a shock and the recovery period are modelling choices, so other choices could give different results.