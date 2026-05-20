"""Unit tests for clustering/clustering_hierarchical.py"""
import pytest
import pandas as pd
import numpy as np

from clustering.clustering_hierarchical import (
    fit_ward,
    fit_bisecting_kmeans,
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


# --- fit_ward ---

def test_fit_ward_label_count(blobs):
    _, labels, _ = fit_ward(blobs, n_clusters=3)
    assert len(labels) == len(blobs)


def test_fit_ward_correct_n_clusters(blobs):
    _, labels, _ = fit_ward(blobs, n_clusters=3)
    assert len(np.unique(labels)) == 3


def test_fit_ward_linkage_shape(blobs):
    Z, _, _ = fit_ward(blobs, n_clusters=3)
    # linkage matrix has n-1 rows and 4 columns
    assert Z.shape == (len(blobs) - 1, 4)


def test_fit_ward_cophenetic_range(blobs):
    _, _, coph = fit_ward(blobs, n_clusters=3)
    assert 0.0 <= coph <= 1.0


def test_fit_ward_cophenetic_high_for_blobs(blobs):
    # well-separated blobs should give high cophenetic correlation
    _, _, coph = fit_ward(blobs, n_clusters=3)
    assert coph > 0.5


# --- fit_bisecting_kmeans ---

def test_fit_bkm_label_count(blobs):
    _, labels = fit_bisecting_kmeans(blobs, n_clusters=3)
    assert len(labels) == len(blobs)


def test_fit_bkm_correct_n_clusters(blobs):
    _, labels = fit_bisecting_kmeans(blobs, n_clusters=3)
    assert len(np.unique(labels)) == 3


def test_fit_bkm_reproducible(blobs):
    _, labels_a = fit_bisecting_kmeans(blobs, n_clusters=3, random_state=0)
    _, labels_b = fit_bisecting_kmeans(blobs, n_clusters=3, random_state=0)
    assert np.array_equal(labels_a, labels_b)


# --- cluster_summary ---

def test_cluster_summary_has_overall(sample_features):
    labels = np.array([0] * 20 + [1] * 20 + [2] * 20)
    summary = cluster_summary(sample_features, labels)
    assert "Overall" in summary.index


def test_cluster_summary_n_drivers(sample_features):
    labels = np.array([0] * 20 + [1] * 20 + [2] * 20)
    summary = cluster_summary(sample_features, labels)
    assert summary.loc["Cluster 0", "n_drivers"] == 20


def test_cluster_summary_overall_count(sample_features):
    labels = np.array([0] * 20 + [1] * 20 + [2] * 20)
    summary = cluster_summary(sample_features, labels)
    assert summary.loc["Overall", "n_drivers"] == len(sample_features)


# --- assign_archetypes ---

def test_assign_archetypes_maps_correctly():
    labels = np.array([0, 1, 2])
    mapping = {0: "A", 1: "B", 2: "C"}
    result = assign_archetypes(labels, archetype_map=mapping)
    assert list(result) == ["A", "B", "C"]


def test_assign_archetypes_unknown_label_fallback():
    labels = np.array([5])
    result = assign_archetypes(labels, archetype_map={})
    assert result[0] == "Cluster 5"
