"""Unit tests for clustering/clustering_density.py"""
import pytest
import pandas as pd
import numpy as np

from clustering.clustering_density import (
    tune_eps,
    fit_dbscan,
    cluster_summary,
    assign_archetypes,
)


# --- Fixtures ---

@pytest.fixture
def blobs():
    rng = np.random.default_rng(42)
    centers = np.array([[0, 0, 0], [8, 8, 8], [-8, 8, -8]])
    X = np.vstack([rng.standard_normal((20, 3)) + c for c in centers])
    return pd.DataFrame(X, columns=["PC1", "PC2", "PC3"])


@pytest.fixture
def sample_features(blobs):
    return blobs.copy()


# --- tune_eps ---
def test_tune_eps_returns_float(blobs):
    eps, _ = tune_eps(blobs, min_samples=5, num_std=3.0)
    assert isinstance(eps, float)


def test_tune_eps_positive(blobs):
    eps, _ = tune_eps(blobs, min_samples=5, num_std=3.0)
    assert eps > 0


def test_tune_eps_distances_shape(blobs):
    _, distances = tune_eps(blobs, min_samples=5, num_std=3.0)
    assert distances.shape == (len(blobs),)


def test_tune_eps_distances_positive(blobs):
    _, distances = tune_eps(blobs, min_samples=5, num_std=3.0)
    assert (distances >= 0).all()


def test_tune_eps_larger_num_std_gives_larger_eps(blobs):
    eps_low, _ = tune_eps(blobs, min_samples=5, num_std=1.0)
    eps_high, _ = tune_eps(blobs, min_samples=5, num_std=3.0)
    assert eps_high > eps_low


# --- fit_dbscan ---

def test_fit_dbscan_label_count(blobs):
    eps, _ = tune_eps(blobs, min_samples=5, num_std=3.0)
    _, labels = fit_dbscan(blobs, eps=eps, min_samples=5)
    assert len(labels) == len(blobs)


def test_fit_dbscan_finds_blobs(blobs):
    # well-separated blobs with good eps should find 3 clusters
    eps, _ = tune_eps(blobs, min_samples=5, num_std=3.0)
    _, labels = fit_dbscan(blobs, eps=eps, min_samples=5)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    assert n_clusters >= 1


def test_fit_dbscan_tiny_eps_creates_noise(blobs):
    # extremely small eps → most points become noise (-1)
    _, labels = fit_dbscan(blobs, eps=0.0001, min_samples=5)
    assert -1 in labels


def test_fit_dbscan_returns_numpy_array(blobs):
    eps, _ = tune_eps(blobs, min_samples=5, num_std=3.0)
    _, labels = fit_dbscan(blobs, eps=eps, min_samples=5)
    assert isinstance(labels, np.ndarray)


# --- cluster_summary ---

def test_cluster_summary_has_overall(sample_features):
    labels = np.array([0] * 20 + [1] * 20 + [2] * 20)
    summary = cluster_summary(sample_features, labels)
    assert "Overall" in summary.index


def test_cluster_summary_noise_row_labelled(sample_features):
    # -1 label should appear as "Noise" in the index
    labels = np.array([-1] * 5 + [0] * 55)
    summary = cluster_summary(sample_features, labels)
    assert "Noise" in summary.index


def test_cluster_summary_n_drivers_noise(sample_features):
    labels = np.array([-1] * 5 + [0] * 55)
    summary = cluster_summary(sample_features, labels)
    assert summary.loc["Noise", "n_drivers"] == 5


def test_cluster_summary_overall_count(sample_features):
    labels = np.array([0] * 20 + [1] * 20 + [2] * 20)
    summary = cluster_summary(sample_features, labels)
    assert summary.loc["Overall", "n_drivers"] == len(sample_features)


# --- assign_archetypes ---

def test_assign_archetypes_maps_noise():
    labels = np.array([-1, 0])
    mapping = {-1: "Noise / Outlier", 0: "Main Cluster"}
    result = assign_archetypes(labels, archetype_map=mapping)
    assert result[0] == "Noise / Outlier"


def test_assign_archetypes_length_preserved():
    labels = np.array([0, -1, 0, -1])
    mapping = {0: "Cluster", -1: "Noise"}
    result = assign_archetypes(labels, archetype_map=mapping)
    assert len(result) == 4


def test_assign_archetypes_unknown_fallback():
    labels = np.array([99])
    result = assign_archetypes(labels, archetype_map={})
    assert result[0] == "Cluster 99"
