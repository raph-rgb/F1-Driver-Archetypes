"""Unit tests for pca.py"""
import pytest
import pandas as pd
import numpy as np

from pca import (
    impute_features,
    scale_features,
    run_pca,
    choose_n_components,
    get_loadings,
)

FEATURE_COLS = [
    "avg_finish_position",
    "avg_grid_position",
    "avg_points_per_race",
    "win_rate",
    "podium_rate",
    "dnf_rate",
    "avg_fastest_lap_speed",
    "avg_best_quali_time",
]


# --- Fixtures ---

@pytest.fixture
def sample_features():
    rng = np.random.default_rng(42)
    df = pd.DataFrame(rng.random((20, 8)), columns=FEATURE_COLS)
    return df


@pytest.fixture
def features_with_nulls():
    rng = np.random.default_rng(42)
    df = pd.DataFrame(rng.random((20, 8)), columns=FEATURE_COLS)
    df.loc[[2, 7], "avg_fastest_lap_speed"] = np.nan
    return df


@pytest.fixture
def scaled_features(sample_features):
    scaled, _ = scale_features(sample_features)
    return scaled


# --- impute_features ---

def test_impute_removes_all_nulls(features_with_nulls):
    imputed = impute_features(features_with_nulls)
    assert imputed.isnull().sum().sum() == 0


def test_impute_preserves_shape(features_with_nulls):
    imputed = impute_features(features_with_nulls)
    assert imputed.shape == features_with_nulls.shape


def test_impute_preserves_columns(features_with_nulls):
    imputed = impute_features(features_with_nulls)
    assert list(imputed.columns) == list(features_with_nulls.columns)


def test_impute_uses_median(features_with_nulls):
    # manually compute median of the non-null values
    expected_median = features_with_nulls["avg_fastest_lap_speed"].median()
    imputed = impute_features(features_with_nulls)
    assert imputed.loc[2, "avg_fastest_lap_speed"] == pytest.approx(expected_median)


# --- scale_features ---

def test_scale_mean_near_zero(sample_features):
    scaled, _ = scale_features(sample_features)
    assert scaled.mean().abs().max() == pytest.approx(0.0, abs=1e-10)


def test_scale_std_near_one(sample_features):
    scaled, _ = scale_features(sample_features)
    # StandardScaler uses ddof=0 (population std), so we match that here
    assert scaled.std(ddof=0).min() == pytest.approx(1.0, abs=1e-5)


def test_scale_preserves_shape(sample_features):
    scaled, _ = scale_features(sample_features)
    assert scaled.shape == sample_features.shape


def test_scale_returns_scaler(sample_features):
    from sklearn.preprocessing import StandardScaler
    _, scaler = scale_features(sample_features)
    assert isinstance(scaler, StandardScaler)


# --- run_pca ---

def test_run_pca_output_shape(scaled_features):
    _, X_pca = run_pca(scaled_features)
    assert X_pca.shape == scaled_features.shape


def test_run_pca_column_names(scaled_features):
    _, X_pca = run_pca(scaled_features)
    expected = [f"PC{i+1}" for i in range(scaled_features.shape[1])]
    assert list(X_pca.columns) == expected


def test_run_pca_variance_sums_to_one(scaled_features):
    pca, _ = run_pca(scaled_features)
    assert pca.explained_variance_ratio_.sum() == pytest.approx(1.0, abs=1e-5)


def test_run_pca_returns_pca_object(scaled_features):
    from sklearn.decomposition import PCA
    pca, _ = run_pca(scaled_features)
    assert isinstance(pca, PCA)


# --- choose_n_components ---

def test_choose_n_components_at_90(scaled_features):
    pca, _ = run_pca(scaled_features)
    n = choose_n_components(pca, threshold=0.90)
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    assert cumvar[n - 1] >= 0.90


def test_choose_n_components_minimum(scaled_features):
    pca, _ = run_pca(scaled_features)
    n = choose_n_components(pca, threshold=0.90)
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    # n should be the smallest number that clears the threshold
    if n > 1:
        assert cumvar[n - 2] < 0.90


def test_choose_n_components_returns_int(scaled_features):
    pca, _ = run_pca(scaled_features)
    n = choose_n_components(pca, threshold=0.90)
    assert isinstance(n, int)


# --- get_loadings ---

def test_get_loadings_shape(scaled_features):
    pca, _ = run_pca(scaled_features)
    n = choose_n_components(pca, threshold=0.90)
    loadings = get_loadings(pca, n)
    assert loadings.shape == (n, len(FEATURE_COLS))


def test_get_loadings_index(scaled_features):
    pca, _ = run_pca(scaled_features)
    loadings = get_loadings(pca, 3)
    assert list(loadings.index) == ["PC1", "PC2", "PC3"]


def test_get_loadings_columns(scaled_features):
    pca, _ = run_pca(scaled_features)
    loadings = get_loadings(pca, 2)
    assert list(loadings.columns) == FEATURE_COLS
