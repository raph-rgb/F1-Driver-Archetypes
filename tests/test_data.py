"""Unit tests for data.py"""
import pytest
import pandas as pd
import numpy as np

from data import (
    lap_time_to_seconds,
    filter_to_era,
    compute_dnf_flags,
    compute_qualifying_pace,
)


# --- Fixtures ---

@pytest.fixture
def sample_results():
    return pd.DataFrame({
        "raceId":           [1, 2, 3, 4],
        "driverId":         [10, 10, 20, 20],
        "positionOrder":    [1, 2, 3, 4],
        "grid":             [1, 2, 3, 4],
        "points":           [25.0, 18.0, 15.0, 12.0],
        "fastestLapSpeed":  [210.0, 208.0, None, 205.0],
        "statusId":         [1, 2, 1, 1],
    })


@pytest.fixture
def sample_races():
    return pd.DataFrame({
        "raceId": [1, 2, 3, 4],
        "year":   [2000, 2005, 1999, 2024],
    })


@pytest.fixture
def sample_status():
    return pd.DataFrame({
        "statusId": [1, 2],
        "status":   ["Finished", "Accident"],
    })


@pytest.fixture
def sample_qualifying():
    return pd.DataFrame({
        "raceId":   [1, 2, 3],
        "driverId": [10, 10, 20],
        "q1":       ["1:26.000", "1:25.000", "1:30.000"],
        "q2":       ["1:25.500", None,        "1:29.000"],
        "q3":       [None,       None,        None],
    })


# --- lap_time_to_seconds ---

def test_lap_time_standard():
    assert lap_time_to_seconds("1:26.572") == pytest.approx(86.572)


def test_lap_time_sub_minute():
    assert lap_time_to_seconds("1:05.001") == pytest.approx(65.001)


def test_lap_time_zero_seconds():
    assert lap_time_to_seconds("0:59.999") == pytest.approx(59.999)


def test_lap_time_none_returns_none():
    assert lap_time_to_seconds(None) is None


def test_lap_time_nan_returns_none():
    assert lap_time_to_seconds(float("nan")) is None


def test_lap_time_bad_string_returns_none():
    assert lap_time_to_seconds("bad") is None


# --- filter_to_era ---

def test_filter_to_era_excludes_out_of_range(sample_results, sample_races):
    # race 3 is year 1999, should be excluded
    filtered = filter_to_era(sample_results, sample_races, 2000, 2024)
    assert 3 not in filtered["raceId"].values


def test_filter_to_era_row_count(sample_results, sample_races):
    filtered = filter_to_era(sample_results, sample_races, 2000, 2024)
    assert len(filtered) == 3


def test_filter_to_era_attaches_year(sample_results, sample_races):
    filtered = filter_to_era(sample_results, sample_races, 2000, 2024)
    assert "year" in filtered.columns


def test_filter_to_era_inclusive_bounds(sample_results, sample_races):
    # races 1 (2000) and 4 (2024) are on the boundary — both should be kept
    filtered = filter_to_era(sample_results, sample_races, 2000, 2024)
    assert set([1, 4]).issubset(set(filtered["raceId"].values))


# --- compute_dnf_flags ---

def test_dnf_flags_finished_is_false(sample_results, sample_races, sample_status):
    era = filter_to_era(sample_results, sample_races, 2000, 2024)
    result = compute_dnf_flags(era, sample_status)
    # raceId 1 has statusId=1 (Finished) → dnf should be False
    assert result.loc[result["raceId"] == 1, "dnf"].values[0] == False


def test_dnf_flags_accident_is_true(sample_results, sample_races, sample_status):
    era = filter_to_era(sample_results, sample_races, 2000, 2024)
    result = compute_dnf_flags(era, sample_status)
    # raceId 2 has statusId=2 (Accident) → dnf should be True
    assert result.loc[result["raceId"] == 2, "dnf"].values[0] == True


def test_dnf_flags_column_exists(sample_results, sample_races, sample_status):
    era = filter_to_era(sample_results, sample_races, 2000, 2024)
    result = compute_dnf_flags(era, sample_status)
    assert "dnf" in result.columns


def test_dnf_flags_boolean_dtype(sample_results, sample_races, sample_status):
    era = filter_to_era(sample_results, sample_races, 2000, 2024)
    result = compute_dnf_flags(era, sample_status)
    assert result["dnf"].dtype == bool


# --- compute_qualifying_pace ---

def test_qualifying_pace_returns_correct_columns(sample_qualifying, sample_races):
    pace = compute_qualifying_pace(sample_qualifying, sample_races, 2000, 2024)
    assert list(pace.columns) == ["driverId", "avg_best_quali_time"]


def test_qualifying_pace_uses_best_of_q1_q2(sample_qualifying, sample_races):
    # driver 10, race 1: q1=86.0s, q2=85.5s → best=85.5
    # driver 10, race 2: q1=85.0s, q2=NaN  → best=85.0
    # avg = (85.5 + 85.0) / 2 = 85.25
    pace = compute_qualifying_pace(sample_qualifying, sample_races, 2000, 2024)
    val = pace.loc[pace["driverId"] == 10, "avg_best_quali_time"].values[0]
    assert val == pytest.approx(85.25)


def test_qualifying_pace_excludes_out_of_era(sample_qualifying, sample_races):
    # race 3 is year 1999 (driver 20) — should be excluded
    pace = compute_qualifying_pace(sample_qualifying, sample_races, 2000, 2024)
    assert 20 not in pace["driverId"].values


def test_qualifying_pace_no_nulls(sample_qualifying, sample_races):
    pace = compute_qualifying_pace(sample_qualifying, sample_races, 2000, 2024)
    assert pace["avg_best_quali_time"].isnull().sum() == 0
