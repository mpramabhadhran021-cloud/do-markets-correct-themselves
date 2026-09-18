import os
import pandas as pd

# File locations
raw = os.path.join("..", "data", "raw")
processed = os.path.join("..", "data", "processed")
os.makedirs(processed, exist_ok=True)


def load_data(name):
    file = os.path.join(raw, name + ".csv")
    data = pd.read_csv(file)
    data["date"] = pd.to_datetime(data["date"])
    return data.set_index("date")


# Load the data
wheat = load_data("wheat_price")
oil_price = load_data("oil_price")
oil_production = load_data("oil_production")
oil_stocks = load_data("oil_stocks")
oil_consumption = load_data("oil_consumption")
copper = load_data("copper_price")
cpi = load_data("cpi")
fedfunds = load_data("fedfunds")

# Fill the one missing CPI observation
cpi["cpi"] = cpi["cpi"].interpolate()

# Use January 2020 as the price index base
cpi_base = cpi.loc["2020-01-01", "cpi"]

# Oil data
oil = oil_price.join(
    [oil_production, oil_stocks, oil_consumption, cpi, fedfunds],
    how="left"
)
oil["wti_price_real"] = oil["wti_price_usd_per_bbl"] * cpi_base / oil["cpi"]
oil.to_csv(os.path.join(processed, "oil_panel.csv"))

# Wheat data
wheat = wheat.join([cpi, fedfunds], how="left")
wheat["wheat_price_real"] = wheat["wheat_price_usd_per_ton"] * cpi_base / wheat["cpi"]
wheat.to_csv(os.path.join(processed, "wheat_panel.csv"))

# Copper data
copper = copper.join([cpi, fedfunds], how="left")
copper["copper_price_real"] = copper["copper_price_usd_per_ton"] * cpi_base / copper["cpi"]
copper.to_csv(os.path.join(processed, "copper_panel.csv"))

# Combine the three real-price series
combined = pd.DataFrame(index=oil.index)
combined["wti_price_real"] = oil["wti_price_real"]
combined["wheat_price_real"] = wheat["wheat_price_real"]
combined["copper_price_real"] = copper["copper_price_real"]
combined["cpi"] = cpi["cpi"]
combined["fedfunds"] = fedfunds["fedfunds"]
combined.to_csv(os.path.join(processed, "combined_prices.csv"))

print("Data preparation complete.")
