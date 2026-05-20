"""
data.py
-------
Loads the five raw CSVs, engineers per-driver features, and returns
a combined DataFrame saved to data/combined/data.csv.

Scope: Formula 1 seasons 2000 - 2024 only.
Only drivers with at least MIN_RACES career starts in that window are kept.

Engineered features (one row per driver):
  - avg_finish_position   : mean finishing order across all races entered
  - avg_grid_position     : mean qualifying grid position
  - avg_points_per_race   : mean championship points earned per race
  - win_rate              : proportion of races finished in P1
  - podium_rate           : proportion of races finished in P1-P3
  - dnf_rate              : proportion of races where status != "Finished"
  - avg_fastest_lap_speed : mean fastest lap speed (km/h) where recorded
  - avg_best_quali_time   : mean best qualifying lap time in seconds
                            (best of Q1/Q2/Q3 per session, then averaged)
"""

import pandas as pd

# Define constants
NULL_MARKER = "\\N"     # dataset uses \N for missing values
MIN_RACES   = 30        # minimum number of races within the defined era
ERA_START   = 2000      # lower bound for season filter
ERA_END     = 2024      # upper bound for season filter


def load_raw_data() -> dict[str, pd.DataFrame]:
    """
    Load all five raw CSVs from data/raw/ into a dictionary of DataFrames.

    Returns
    -------
    dict with keys: 'results', 'drivers', 'races', 'qualifying', 'status'
    """
    return {
        "results":    pd.read_csv("data/raw/results.csv",    na_values=NULL_MARKER, low_memory=False),
        "drivers":    pd.read_csv("data/raw/drivers.csv",    na_values=NULL_MARKER, low_memory=False),
        "races":      pd.read_csv("data/raw/races.csv",      na_values=NULL_MARKER, low_memory=False),
        "qualifying": pd.read_csv("data/raw/qualifying.csv", na_values=NULL_MARKER, low_memory=False),
        "status":     pd.read_csv("data/raw/status.csv",     na_values=NULL_MARKER, low_memory=False),
    }


def lap_time_to_seconds(time_str) -> float | None:
    """
    Convert a lap-time string in "M:SS.mmm" format to total seconds.

    Parameters
    ----------
    time_str : str or NaN
        A lap time string formatted as "minutes:seconds.milliseconds".

    Returns
    -------
    float or None
        Total seconds, or None if the input is missing or unparseable.
    """
    if pd.isna(time_str):
        return None
    try:
        parts = str(time_str).strip().split(":")
        minutes = int(parts[0])
        seconds = float(parts[1])
        return minutes * 60 + seconds
    except (IndexError, ValueError):
        return None


# Feature engineering
def filter_to_era(
    results: pd.DataFrame,
    races: pd.DataFrame,
    era_start: int = ERA_START,
    era_end: int = ERA_END,
) -> pd.DataFrame:
    """
    Restrict results to races that took place within the target era.

    Parameters
    ----------
    results   : raw results DataFrame
    races     : raw races DataFrame (contains the 'year' column)
    era_start : first season to include (inclusive)
    era_end   : last season to include (inclusive)

    Returns
    -------
    results DataFrame filtered to the target era, with a 'year' column attached.
    """
    races_in_era = races[races["year"].between(era_start, era_end)][["raceId", "year"]]
    return results.merge(races_in_era, on="raceId", how="inner")


def compute_dnf_flags(results: pd.DataFrame, status: pd.DataFrame) -> pd.DataFrame:
    """
    Attach a boolean 'dnf' column to results.

    Any status other than "Finished" counts as a DNF. This includes
    accidents, mechanical failures, collisions, and disqualifications.

    Parameters
    ----------
    results : era-filtered results DataFrame
    status  : raw status DataFrame

    Returns
    -------
    results DataFrame with an added boolean column 'dnf'.
    """
    results = results.merge(status[["statusId", "status"]], on="statusId", how="left")
    results["dnf"] = results["status"] != "Finished"
    return results


def compute_qualifying_pace(
    qualifying: pd.DataFrame,
    races: pd.DataFrame,
    era_start: int = ERA_START,
    era_end: int = ERA_END,
) -> pd.DataFrame:
    """
    Compute each driver's average best qualifying lap time in seconds,
    restricted to the target era.

    For each session row:
      1. Convert Q1, Q2, Q3 strings to seconds.
      2. Take the best (minimum) of Q1/Q2/Q3, which is the driver's
         fastest qualifying effort that weekend.
         Note: not all drivers reach Q2/Q3, so using the best available
         is the fairest cross-grid comparison.
      3. Average that best time over all sessions per driver.

    Parameters
    ----------
    qualifying : raw qualifying DataFrame
    races      : raw races DataFrame
    era_start  : first season to include
    era_end    : last season to include

    Returns
    -------
    DataFrame with columns ['driverId', 'avg_best_quali_time']
    """
    races_in_era = races[races["year"].between(era_start, era_end)][["raceId"]]
    quali = qualifying.merge(races_in_era, on="raceId", how="inner").copy()

    for col in ["q1", "q2", "q3"]:
        quali[f"{col}_sec"] = quali[col].apply(lap_time_to_seconds)

    quali["best_lap_sec"] = quali[["q1_sec", "q2_sec", "q3_sec"]].min(axis=1)

    return (
        quali.groupby("driverId")["best_lap_sec"]
        .mean()
        .reset_index()
        .rename(columns={"best_lap_sec": "avg_best_quali_time"})
    )


def build_driver_features(
    results: pd.DataFrame,
    drivers: pd.DataFrame,
    races: pd.DataFrame,
    qualifying: pd.DataFrame,
    status: pd.DataFrame,
    min_races: int = MIN_RACES,
    era_start: int = ERA_START,
    era_end: int = ERA_END,
) -> pd.DataFrame:
    """
    Build the one-row-per-driver feature table.

    Parameters
    ----------
    results    : raw results DataFrame
    drivers    : raw drivers DataFrame
    races      : raw races DataFrame
    qualifying : raw qualifying DataFrame
    status     : raw status DataFrame
    min_races  : minimum career starts within the era to keep a driver
    era_start  : first season to include
    era_end    : last season to include

    Returns
    -------
    Clean feature DataFrame that contains one row per driver, all engineered columns.
    """
    # Restrict to the target era and attach DNF flags
    results_era = filter_to_era(results, races, era_start, era_end)
    results_era = compute_dnf_flags(results_era, status)

    # Aggregate per driver
    agg = (
        results_era.groupby("driverId")
        .agg(
            total_races         = ("raceId",         "count"),
            avg_finish_position = ("positionOrder",  "mean"),
            avg_grid_position   = ("grid",           "mean"),
            avg_points_per_race = ("points",         "mean"),
            wins                = ("positionOrder",  lambda x: (x == 1).sum()),
            podiums             = ("positionOrder",  lambda x: (x <= 3).sum()),
            dnfs                = ("dnf",            "sum"),
            avg_fastest_lap_speed = ("fastestLapSpeed", "mean"),
        )
        .reset_index()
    )

    agg["win_rate"]    = agg["wins"]    / agg["total_races"]
    agg["podium_rate"] = agg["podiums"] / agg["total_races"]
    agg["dnf_rate"]    = agg["dnfs"]    / agg["total_races"]

    # Join qualifying pace (era-filtered inside the function)
    quali_pace = compute_qualifying_pace(qualifying, races, era_start, era_end)
    agg = agg.merge(quali_pace, on="driverId", how="left")

    # Join driver name and nationality for labelling
    drivers["driver_name"] = drivers["forename"] + " " + drivers["surname"]
    agg = agg.merge(
        drivers[["driverId", "driver_name", "nationality"]],
        on="driverId",
        how="left",
    )

    # Drop drivers below the minimum race threshold
    agg = agg[agg["total_races"] >= min_races].reset_index(drop=True)

    return agg[[
        "driverId",
        "driver_name",
        "nationality",
        "total_races",
        "avg_finish_position",
        "avg_grid_position",
        "avg_points_per_race",
        "win_rate",
        "podium_rate",
        "dnf_rate",
        "avg_fastest_lap_speed",
        "avg_best_quali_time",
    ]]


# Pipeline entry point
def build_combined_dataset(
    min_races: int = MIN_RACES,
    era_start: int = ERA_START,
    era_end:   int = ERA_END,
) -> pd.DataFrame:
    """
    Full pipeline: load raw CSVs -> engineer features -> save data.csv.

    Reads from  : data/raw/
    Writes to   : data/combined/data.csv

    Parameters
    ----------
    min_races : minimum career starts within the era to include a driver
    era_start : first season to include
    era_end   : last season to include

    Returns
    -------
    The final feature DataFrame (also saved to data/combined/data.csv).
    """
    print("Loading raw data...")
    raw = load_raw_data()

    print(f"Engineering features (era: {era_start}–{era_end}, min races: {min_races})...")
    df = build_driver_features(
        results=raw["results"],
        drivers=raw["drivers"],
        races=raw["races"],
        qualifying=raw["qualifying"],
        status=raw["status"],
        min_races=min_races,
        era_start=era_start,
        era_end=era_end,
    )

    output_path = "data/combined/data.csv"
    df.to_csv(output_path, index=False)
    print(f"Saved to {output_path}  |  shape: {df.shape}")
    return df


# Run script
if __name__ == "__main__":
    df = build_combined_dataset()
    print("\nSample rows:")
    print(df.head(10).to_string())
    print("\nNull counts:")
    print(df.isnull().sum())
    print("\nBasic stats:")
    print(df.describe().to_string())
