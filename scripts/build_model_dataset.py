from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd


FAF5_URL = "https://github.com/bnn05195/data-science/releases/download/v1.0/FAF5.parquet"

STATE_CENTROIDS = {
    1: (32.8, -86.7),
    2: (61.3, -152.4),
    4: (34.1, -111.9),
    5: (34.9, -92.3),
    6: (36.1, -119.6),
    8: (39.0, -105.7),
    9: (41.5, -72.7),
    10: (39.3, -75.5),
    11: (38.9, -77.0),
    12: (27.7, -81.6),
    13: (33.0, -83.6),
    15: (21.0, -157.8),
    16: (44.2, -114.4),
    17: (40.3, -88.9),
    18: (39.8, -86.2),
    19: (42.0, -93.4),
    20: (38.5, -98.3),
    21: (37.6, -84.6),
    22: (31.1, -91.8),
    23: (44.6, -69.3),
    24: (39.0, -76.8),
    25: (42.2, -71.8),
    26: (43.3, -84.5),
    27: (45.6, -93.9),
    28: (32.7, -89.6),
    29: (38.4, -92.2),
    30: (46.9, -110.4),
    31: (41.1, -98.2),
    32: (38.3, -117.0),
    33: (43.4, -71.5),
    34: (40.2, -74.5),
    35: (34.8, -106.2),
    36: (42.1, -74.9),
    37: (35.6, -79.8),
    38: (47.5, -100.5),
    39: (40.3, -82.7),
    40: (35.5, -96.9),
    41: (44.5, -122.1),
    42: (40.5, -77.2),
    44: (41.6, -71.5),
    45: (33.8, -80.9),
    46: (44.2, -99.4),
    47: (35.7, -86.6),
    48: (31.0, -97.5),
    49: (40.1, -111.8),
    50: (44.0, -72.7),
    51: (37.7, -78.1),
    53: (47.4, -121.4),
    54: (38.4, -80.9),
    55: (44.2, -89.6),
    56: (42.7, -107.3),
}


def haversine_miles(origin_state: int, dest_state: int) -> float:
    """Approximate state-to-state distance using state centroid coordinates."""
    if origin_state == dest_state:
        return 0.0
    if origin_state not in STATE_CENTROIDS or dest_state not in STATE_CENTROIDS:
        return np.nan

    lat1, lon1 = STATE_CENTROIDS[origin_state]
    lat2, lon2 = STATE_CENTROIDS[dest_state]
    radius_miles = 3958.8

    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * radius_miles * math.asin(math.sqrt(a))


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_state_names(data_dir: Path) -> pd.DataFrame:
    state = pd.read_excel(data_dir / "FAF5_metadata.xlsx", sheet_name="State")
    state = state.rename(columns={"Numeric Label": "state_code", "Description": "state_name"})
    state = state[["state_code", "state_name"]].dropna()
    state["state_code"] = state["state_code"].astype(int)
    return state


def load_annual_cpi(data_dir: Path) -> pd.DataFrame:
    cpi = pd.read_csv(data_dir / "CPIAUCSL_PC1.csv")
    cpi["year"] = pd.to_datetime(cpi["observation_date"], errors="coerce").dt.year
    cpi["cpi_pc1"] = pd.to_numeric(cpi["CPIAUCSL_PC1"], errors="coerce")
    return (
        cpi.dropna(subset=["year", "cpi_pc1"])
        .groupby("year", as_index=False)["cpi_pc1"]
        .mean()
    )


def build_od_year_dataset(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    faf = pd.read_parquet(FAF5_URL)

    tons_cols = sorted(c for c in faf.columns if c.startswith("tons_"))
    value_cols = sorted(c for c in faf.columns if c.startswith("value_"))
    required_cols = ["dms_mode", "dms_orig", "dms_dest"] + tons_cols + value_cols
    missing_cols = [col for col in required_cols if col not in faf.columns]
    if missing_cols:
        raise ValueError(f"FAF5 data is missing required columns: {missing_cols}")

    truck = faf.loc[faf["dms_mode"] == 1, required_cols].dropna(subset=["dms_orig", "dms_dest"])
    truck = truck.copy()
    truck["origin_state"] = (truck["dms_orig"] // 10).astype(int)
    truck["dest_state"] = (truck["dms_dest"] // 10).astype(int)
    truck = truck[
        truck["origin_state"].isin(STATE_CENTROIDS)
        & truck["dest_state"].isin(STATE_CENTROIDS)
    ]

    tons_long = truck.melt(
        id_vars=["origin_state", "dest_state"],
        value_vars=tons_cols,
        var_name="year",
        value_name="tons",
    )
    tons_long["year"] = tons_long["year"].str.replace("tons_", "", regex=False).astype(int)

    value_long = truck.melt(
        id_vars=["origin_state", "dest_state"],
        value_vars=value_cols,
        var_name="year",
        value_name="value",
    )
    value_long["year"] = value_long["year"].str.replace("value_", "", regex=False).astype(int)

    tons_by_od_year = (
        tons_long.groupby(["origin_state", "dest_state", "year"], as_index=False)["tons"]
        .sum()
    )
    value_by_od_year = (
        value_long.groupby(["origin_state", "dest_state", "year"], as_index=False)["value"]
        .sum()
    )

    od_year = (
        tons_by_od_year.merge(value_by_od_year, on=["origin_state", "dest_state", "year"], how="left")
        .fillna({"value": 0})
    )

    state_names = load_state_names(data_dir)
    od_year = od_year.merge(
        state_names.rename(columns={"state_code": "origin_state", "state_name": "origin_state_name"}),
        on="origin_state",
        how="left",
    )
    od_year = od_year.merge(
        state_names.rename(columns={"state_code": "dest_state", "state_name": "dest_state_name"}),
        on="dest_state",
        how="left",
    )

    cpi = load_annual_cpi(data_dir)
    od_year = od_year.merge(cpi, on="year", how="left")

    od_year["distance_miles"] = [
        haversine_miles(origin, dest)
        for origin, dest in zip(od_year["origin_state"], od_year["dest_state"])
    ]
    od_year = od_year.sort_values(["origin_state", "dest_state", "year"]).reset_index(drop=True)
    od_year["tons_lag1"] = od_year.groupby(["origin_state", "dest_state"])["tons"].shift(1)
    od_year["value_lag1"] = od_year.groupby(["origin_state", "dest_state"])["value"].shift(1)
    od_year["tons_growth"] = (od_year["tons"] - od_year["tons_lag1"]) / od_year["tons_lag1"].replace(0, np.nan)
    od_year["value_growth"] = (od_year["value"] - od_year["value_lag1"]) / od_year["value_lag1"].replace(0, np.nan)
    od_year[["tons_growth", "value_growth"]] = (
        od_year[["tons_growth", "value_growth"]]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
    )
    od_year["target_next_year_tons"] = od_year.groupby(["origin_state", "dest_state"])["tons"].shift(-1)

    feature_cols = [
        "origin_state",
        "origin_state_name",
        "dest_state",
        "dest_state_name",
        "year",
        "tons",
        "value",
        "tons_lag1",
        "value_lag1",
        "tons_growth",
        "value_growth",
        "distance_miles",
        "cpi_pc1",
        "target_next_year_tons",
    ]
    od_year = od_year[feature_cols]

    training = od_year.dropna(
        subset=["tons", "tons_lag1", "distance_miles", "cpi_pc1", "target_next_year_tons"]
    ).copy()
    latest_features = od_year[od_year["year"] == od_year["year"].max()].drop(
        columns=["target_next_year_tons"]
    )
    return training, latest_features


def main() -> None:
    parser = argparse.ArgumentParser(description="Build OD-year training data for truck demand forecasting.")
    parser.add_argument("--output", default="data/model_dataset.csv", help="Path for training dataset CSV.")
    parser.add_argument(
        "--latest-output",
        default="data/latest_features.csv",
        help="Path for latest-year feature CSV used for future prediction.",
    )
    args = parser.parse_args()

    root = repo_root()
    data_dir = root / "data"
    output_path = root / args.output
    latest_output_path = root / args.latest_output

    training, latest_features = build_od_year_dataset(data_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    latest_output_path.parent.mkdir(parents=True, exist_ok=True)
    training.to_csv(output_path, index=False, encoding="utf-8-sig")
    latest_features.to_csv(latest_output_path, index=False, encoding="utf-8-sig")

    print(f"saved training dataset: {output_path} rows={len(training):,} cols={len(training.columns)}")
    print(f"saved latest features: {latest_output_path} rows={len(latest_features):,} cols={len(latest_features.columns)}")
    print(f"years in training dataset: {training['year'].min()}-{training['year'].max()}")


if __name__ == "__main__":
    main()
