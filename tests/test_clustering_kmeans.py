"""Unit tests for clustering/clustering_kmeans.py"""
import pytest
import pandas as pd
import numpy as np

from clustering.clustering_kmeans import (
    tune_kmeans,
    fit_kmeans,
    cluster_summary,
    assign_archetypes,
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
def blobs():
    # 3 clearly separated blobs — clustering should find k=3 cleanly
    rng = np.random.default_rng(42)
    centers = np.array([[0, 0, 0], [8, 8, 8], [-8, 8, -8]])
    X = np.vstack([rng.standard_normal((20, 3)) + c for c in centers])
    return pd.DataFrame(X, columns=["PC1", "PC2", "PC3"])


@pytest.fixture
def sample_features():
    rng = np.random.default_rng(0)
    return pd.DataFrame(rng.random((60, len(FEATURE_COLS))), columns=FEATURE_COLS)


# --- tune_kmeans ---

def test_tune_kmeans_returns_all_k(blobs):
    result = tune_kmeans(blobs, k_range=range(2, 5), num_trials=3)
    assert result["k_range"] == [2, 3, 4]


def test_tune_kmeans_inertia_decreasing(blobs):
    result = tune_kmeans(blobs, k_range=range(2, 6), num_trials=3)
    # inertia should decrease as k increases
    assert all(
        result["inertia_mean"][i] >= result["inertia_mean"][i + 1]
        for i in range(len(result["inertia_mean"]) - 1)
    )


def test_tune_kmeans_ch_starts_at_2(blobs):
    result = tune_kmeans(blobs, k_range=range(2, 5), num_trials=3)
    assert result["multi_k_range"][0] == 2


def test_tune_kmeans_finds_3_blobs(blobs):
    result = tune_kmeans(blobs, k_range=range(2, 6), num_trials=3)
    best_k = result["multi_k_range"][np.argmax(result["ch_mean"])]
    assert best_k == 3


def test_tune_kmeans_array_lengths_match(blobs):
    result = tune_kmeans(blobs, k_range=range(2, 5), num_trials=3)
    assert len(result["inertia_mean"]) == len(result["k_range"])
    assert len(result["ch_mean"]) == len(result["multi_k_range"])


# --- fit_kmeans ---

def test_fit_kmeans_label_count(blobs):
    _, labels, _ = fit_kmeans(blobs, k=3, num_trials=3)
    assert len(labels) == len(blobs)


def test_fit_kmeans_correct_n_clusters(blobs):
    _, labels, _ = fit_kmeans(blobs, k=3, num_trials=3)
    assert len(np.unique(labels)) == 3


def test_fit_kmeans_centers_shape(blobs):
    _, _, centers = fit_kmeans(blobs, k=3, num_trials=3)
    assert centers.shape == (3, blobs.shape[1])


def test_fit_kmeans_centers_columns(blobs):
    _, _, centers = fit_kmeans(blobs, k=3, num_trials=3)
    assert list(centers.columns) == list(blobs.columns)


# --- cluster_summary ---

def test_cluster_summary_has_overall_row(sample_features):
    labels = np.array([0] * 20 + [1] * 20 + [2] * 20)
    summary = cluster_summary(sample_features, labels)
    assert "Overall" in summary.index


def test_cluster_summary_n_drivers_correct(sample_features):
    labels = np.array([0] * 20 + [1] * 20 + [2] * 20)
    summary = cluster_summary(sample_features, labels)
    assert summary.loc["Cluster 0", "n_drivers"] == 20


def test_cluster_summary_overall_n_drivers(sample_features):
    labels = np.array([0] * 20 + [1] * 20 + [2] * 20)
    summary = cluster_summary(sample_features, labels)
    assert summary.loc["Overall", "n_drivers"] == len(sample_features)


def test_cluster_summary_row_count(sample_features):
    labels = np.array([0] * 30 + [1] * 30)
    summary = cluster_summary(sample_features, labels)
    # 2 clusters + 1 Overall row
    assert len(summary) == 3


# --- assign_archetypes ---

def test_assign_archetypes_custom_map():
    labels = np.array([0, 1, 2, 0])
    mapping = {0: "Champions", 1: "Midfield", 2: "Backmarkers"}
    result = assign_archetypes(labels, archetype_map=mapping)
    assert list(result) == ["Champions", "Midfield", "Backmarkers", "Champions"]


def test_assign_archetypes_default_fallback():
    # label 99 not in map → falls back to "Cluster 99"
    labels = np.array([99])
    result = assign_archetypes(labels, archetype_map={})
    assert result[0] == "Cluster 99"


def test_assign_archetypes_length_preserved():
    labels = np.array([0, 1, 2, 1, 0])
    result = assign_archetypes(labels, archetype_map={0: "A", 1: "B", 2: "C"})
    assert len(result) == 5
