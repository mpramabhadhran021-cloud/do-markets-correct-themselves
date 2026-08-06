# Do Markets Correct Themselves?
### A Statistical Investigation of Adam Smith's Invisible Hand Using Real-World Market Data

## Overview

This project asks: **do competitive markets show statistical evidence of self-correction
after supply and demand changes?** It studies three real markets — crude oil (WTI),
wheat, and copper — using real, publicly sourced monthly data from 1986/1992–2026.

## Project structure

```
project/
├── data/
│   ├── raw/            # Raw series exactly as collected from source (FRED / EIA)
│   └── processed/       # Cleaned, merged, inflation-adjusted panels (data_prep.py output)
├── scripts/
│   ├── data_prep.py     # Reproducible cleaning/merging pipeline (raw -> processed)
│   ├── nb_builder.py    # Helper used to programmatically build the notebooks
│   └── build_nb*.py     # Scripts that generate the two notebooks (for transparency)
└── notebooks/
    ├── 01_market_data_analysis.ipynb              # Data, EDA, core regressions, lag/ACF analysis
    └── 02_market_self_correction_analysis.ipynb   # Shocks, bootstrap, robustness, cross-market, conclusion
```

## Reproducing the analysis

```bash
cd scripts
python3 data_prep.py                 # rebuilds data/processed/*.csv from data/raw/*.csv
jupyter nbconvert --to notebook --execute --inplace ../notebooks/01_market_data_analysis.ipynb
jupyter nbconvert --to notebook --execute --inplace ../notebooks/02_market_self_correction_analysis.ipynb
```
Requires: `pandas numpy scipy statsmodels matplotlib seaborn`.

## Data sources

| Series | Source | Series ID | Span |
|---|---|---|---|
| WTI crude oil price | EIA via FRED | MCOILWTICO | 1986–2026 |
| U.S. crude oil production | EIA | MCRFPUS2 | 1986–2026 |
| U.S. crude oil ending stocks | EIA | MCESTUS1 | 2005–2026 |
| U.S. petroleum product supplied | EIA | MTTUPUS1 | 1986–2026 |
| Global wheat price | IMF via FRED | PWHEAMTUSDM | 1992–2026 |
| Global copper price | IMF via FRED | PCOPPUSDM | 1992–2026 |
| U.S. CPI | BLS via FRED | CPIAUCSL | 1986–2026 |
| Federal funds rate | Federal Reserve via FRED | FEDFUNDS | 1986–2026 |

All series retrieved from https://fred.stlouisfed.org and https://www.eia.gov, July–August 2026.
Full citations are given in Notebook 1's introduction. IMF-sourced series (wheat, copper) are
copyrighted by the IMF and reproduced under FRED's citation terms for non-commercial academic use.

## Headline findings

- **H1 (supply–price link):** Oil production growth and price growth are significantly
  *negatively* related — but only once both series are correctly treated as non-stationary
  and analyzed in first differences. A naive levels regression gives the wrong sign.
- **H2 (price → future behavior):** Oil producers raise output 3–12 months after price
  increases (significant). U.S. consumption does not respond significantly at these
  horizons (consistent with known short-run demand inelasticity).
- **H3 (adjustment after shocks):** All three markets show statistically significant mean
  reversion in real prices (bootstrap-confirmed), though several major shocks (2008,
  2014–16 oil; 2022 wheat; 2021–22 copper) had not fully reverted to pre-shock levels
  within a 24-month window.
- **H4 (adjustment speed differs by market):** Oil reverts fastest (~4–5 month half-life),
  wheat intermediate (~5 months), copper slowest (~8 months) — robust across sample splits
  and window choices, and consistent with market structure (deep/liquid oil market with
  fast swing producers vs. geographically concentrated, slow-to-expand copper supply).

See Notebook 2's Conclusion for the full, balanced discussion, including limitations.
