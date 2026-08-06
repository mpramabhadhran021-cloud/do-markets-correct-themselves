"""
Data preparation pipeline for "Do Markets Correct Themselves?"
Reads raw series (collected from FRED / EIA, sourced from IMF Primary Commodity
Prices and U.S. Energy Information Administration) and builds three clean,
merged monthly panels: oil, wheat, copper.

Run: python3 data_prep.py
Inputs:  ../data/raw/*.csv
Outputs: ../data/processed/*.csv
"""
import pandas as pd
import numpy as np
import os

RAW = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
OUT = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
os.makedirs(OUT, exist_ok=True)


def load(name, parse_dates=True):
    df = pd.read_csv(os.path.join(RAW, f"{name}.csv"))
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date")


def main():
    wheat = load("wheat_price")
    oil_p = load("oil_price")
    oil_prod = load("oil_production")
    oil_stk = load("oil_stocks")
    oil_cons = load("oil_consumption")
    copper = load("copper_price")
    cpi = load("cpi")
    ff = load("fedfunds")

    # CPI has one missing observation (2025-10-01); linearly interpolate --
    # this is the only missing value in any macro series and sits in the interior
    # of the sample, so interpolation is safe and standard practice.
    cpi["cpi"] = cpi["cpi"].interpolate(method="linear")

    cpi_base = cpi.loc["2020-01-01", "cpi"]  # express real prices in Jan-2020 dollars

    # ---------------- OIL PANEL ----------------
    oil = oil_p.join([oil_prod, oil_stk, oil_cons, cpi, ff], how="left")
    oil["wti_price_real"] = oil["wti_price_usd_per_bbl"] * cpi_base / oil["cpi"]
    oil = oil.sort_index()
    oil.to_csv(os.path.join(OUT, "oil_panel.csv"))

    # ---------------- WHEAT PANEL ----------------
    wheatp = wheat.join([cpi, ff], how="left")
    wheatp["wheat_price_real"] = wheatp["wheat_price_usd_per_ton"] * cpi_base / wheatp["cpi"]
    wheatp = wheatp.sort_index()
    wheatp.to_csv(os.path.join(OUT, "wheat_panel.csv"))

    # ---------------- COPPER PANEL ----------------
    copperp = copper.join([cpi, ff], how="left")
    copperp["copper_price_real"] = copperp["copper_price_usd_per_ton"] * cpi_base / copperp["cpi"]
    copperp = copperp.sort_index()
    copperp.to_csv(os.path.join(OUT, "copper_panel.csv"))

    # ---------------- COMBINED (for cross-market work) ----------------
    combined = pd.DataFrame(index=oil.index)
    combined["wti_price_real"] = oil["wti_price_real"]
    combined = combined.join(wheatp["wheat_price_real"])
    combined = combined.join(copperp["copper_price_real"])
    combined = combined.join(cpi)
    combined = combined.join(ff)
    combined.to_csv(os.path.join(OUT, "combined_prices.csv"))

    # ---------------- Missingness report ----------------
    report = {
        "oil_panel": oil.isna().sum().to_dict(),
        "wheat_panel": wheatp.isna().sum().to_dict(),
        "copper_panel": copperp.isna().sum().to_dict(),
    }
    print("Rows: oil=%d wheat=%d copper=%d combined=%d" % (
        len(oil), len(wheatp), len(copperp), len(combined)))
    for k, v in report.items():
        print(k, v)


if __name__ == "__main__":
    main()
