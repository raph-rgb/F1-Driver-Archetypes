"""Unit tests for clustering.py"""
import pytest
import numpy as np
import pandas as pd

from src.clustering import FEATURE_COLS


def _sample_X(n=60, p=3) -> np.ndarray:
    rng = np.random.default_rng(42)
    # 3 clearly separated blobs
    centers = np.array([[0, 0, 0], [10, 10, 10], [-10, 10, -10]])
    X = np.vstack([
        rng.standard_normal((n // 3, p)) + c for c in centers
    ])
    return X


def test_elbow_scores_returns_all_k():
    from src.clustering import elbow_scores
    X = _sample_X()
    scores = elbow_scores(X, k_range=range(2, 6))
    assert list(scores["k"]) == [2, 3, 4, 5]


def test_elbow_scores_columns():
    from src.clustering import elbow_scores
    X = _sample_X()
    scores = elbow_scores(X, k_range=range(2, 4))
    assert set(scores.columns) == {"k", "inertia", "silhouette_score"}


def test_best_k_finds_correct_cluster_count():
    from src.clustering import elbow_scores, best_k
    X = _sample_X()
    scores = elbow_scores(X, k_range=range(2, 7))
    k = best_k(scores)
    assert k == 3  # data has 3 blobs


def test_run_kmeans_label_count():
    from src.clustering import run_kmeans
    X = _sample_X()
    labels, model = run_kmeans(X, k=3)
    assert len(labels) == len(X)
    assert len(set(labels)) == 3


def test_run_dbscan_returns_array():
    from src.clustering import run_dbscan
    X = _sample_X()
    labels = run_dbscan(X, eps=2.0, min_samples=3)
    assert isinstance(labels, np.ndarray)
    assert len(labels) == len(X)


def test_assign_archetypes_custom_map():
    from src.clustering import assign_archetypes
    labels = np.array([0, 1, 2, 0])
    mapping = {0: "Elite", 1: "Mid", 2: "Back"}
    archetypes = assign_archetypes(labels, archetype_map=mapping)
    assert list(archetypes) == ["Elite", "Mid", "Back", "Elite"]


def test_cluster_summary_shape():
    from src.clustering import cluster_summary
    n = 30
    rng = np.random.default_rng(0)
    df = pd.DataFrame(rng.random((n, len(FEATURE_COLS))), columns=FEATURE_COLS)
    labels = np.array([0] * 10 + [1] * 10 + [2] * 10)
    summary = cluster_summary(df, labels)
    assert len(summary) == 3
    assert "n_drivers" in summary.columns


def test_feature_cols_match_raphs_dataset():
    """Confirm FEATURE_COLS matches the columns in data/combined/data.csv."""
    expected = [
        "avg_finish_position",
        "avg_grid_position",
        "avg_points_per_race",
        "win_rate",
        "podium_rate",
        "dnf_rate",
        "avg_fastest_lap_speed",
        "avg_best_quali_time",
    ]
    assert FEATURE_COLS == expected
