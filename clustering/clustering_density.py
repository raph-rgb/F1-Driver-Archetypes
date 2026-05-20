"""
clustering/clustering_density.py
---------------------------------
Density-based clustering (DBSCAN) for the F1 driver performance project.

DBSCAN groups points that are closely packed together and marks points in
low-density regions as noise (-1). Unlike K-Means it does not require k
upfront and can find arbitrarily shaped clusters.

Important finding: our PCA-transformed driver data does not contain
density-separated regions — DBSCAN returns a single cluster with a small
number of outliers across all reasonable eps values. This is a meaningful
result: driver performance exists on a continuous gradient rather than
in isolated density islands. This finding supports and contextualises
the K-Means and hierarchical cluster boundaries.

Functions
---------
- tune_eps()           - computes k-NN distances and z-score threshold to
                         determine eps automatically, following the lecture
- fit_dbscan()         - fits DBSCAN with given eps and min_samples
- cluster_summary()    - mean feature values per cluster (noise as -1)
- assign_archetypes()  - maps cluster integers to readable name strings

Plot functions (call from report.ipynb)
---------------------------------------
- plot_neighbor_distances() - re-exported from utils for convenience
"""

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
from clustering.utils import (
    plot_clustering,
    plot_cluster_averages,
    plot_neighbor_distances,
)


# Default min_samples: ~10% of dataset size following lecture convention
MIN_SAMPLES = 7   # 10% of 77 drivers, rounded

# fill in after inspecting cluster_summary() — remove if not needed
ARCHETYPE_MAP: dict[int, str] = {
    0: "Cluster 0",
    -1: "Noise / Outlier",
}


# Compute eps from k-NN distances using z-score threshold
def tune_eps(
    X: pd.DataFrame,
    min_samples: int = MIN_SAMPLES,
    num_std: float = 3.0,
) -> tuple[float, np.ndarray]:
    """
    Estimate eps for DBSCAN using the tau-nearest neighbor distance method.

    For each point, compute the distance to its min_samples-th nearest
    neighbour. Points inside dense clusters will have small values; noise
    points will have large values. The threshold eps = mean + num_std * std
    sits at the boundary of the tail, separating core from noise.

    Parameters
    ----------
    X           : feature DataFrame (PCA-transformed coordinates).
    min_samples : number of neighbours to use (the tau parameter).
    num_std     : z-score multiplier for the threshold (default 3.0).

    Returns
    -------
    eps       : float, recommended epsilon value.
    distances : 1-D array of tau-nearest neighbor distances (for plotting).
    """
    knn = NearestNeighbors(n_neighbors=min_samples).fit(X)
    dist_matrix, _ = knn.kneighbors(X)
    distances = dist_matrix[:, min_samples - 1]
    eps = distances.mean() + num_std * distances.std()
    print(f"Recommended eps: {eps:.4f}  (min_samples={min_samples}, num_std={num_std})")
    return eps, distances


# Fit DBSCAN
def fit_dbscan(
    X: pd.DataFrame,
    eps: float,
    min_samples: int = MIN_SAMPLES,
) -> tuple[DBSCAN, np.ndarray]:
    """
    Fit DBSCAN with given eps and min_samples.

    Labels of -1 indicate noise points — drivers that do not belong to
    any dense cluster. These are statistically interesting outliers worth
    calling out in the report.

    Parameters
    ----------
    X           : feature DataFrame to cluster.
    eps         : neighbourhood radius (use tune_eps to set this).
    min_samples : minimum points to form a dense region.

    Returns
    -------
    model  : fitted DBSCAN object.
    labels : cluster label array (-1 = noise).
    """
    model = DBSCAN(eps=eps, min_samples=min_samples)
    labels = model.fit_predict(X)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = (labels == -1).sum()
    print(f"Clusters found: {n_clusters}  |  Noise points: {n_noise}")
    return model, labels


# Mean feature values per cluster (noise included as -1)
def cluster_summary(
    features: pd.DataFrame,
    labels: np.ndarray,
) -> pd.DataFrame:
    """
    Mean of each feature per cluster, with noise (-1) as its own row
    and overall mean as a reference row.

    Use this to see whether noise points share common characteristics
    (e.g. unusually dominant champions or extreme backmarkers).

    Parameters
    ----------
    features : original (pre-PCA) feature DataFrame, one row per driver.
    labels   : cluster label array from fit_dbscan (-1 = noise).

    Returns
    -------
    DataFrame — one row per cluster/noise group plus an 'Overall' row.
    """
    df = features.copy()
    df["cluster"] = labels
    summary = df.groupby("cluster").mean()
    summary["n_drivers"] = df.groupby("cluster").size()

    def label_index(i):
        return "Noise" if i == -1 else f"Cluster {i}"

    summary.index = [label_index(i) for i in summary.index]
    overall = features.mean().to_frame().T
    overall.index = ["Overall"]
    overall["n_drivers"] = len(features)
    return pd.concat([summary, overall])


# Map cluster integers to readable names
def assign_archetypes(
    labels: np.ndarray,
    archetype_map: dict | None = None,
) -> np.ndarray:
    """
    Map cluster integers to human-readable archetype names.

    Fill in ARCHETYPE_MAP at the top of this file after inspecting
    cluster_summary(). If you decide not to use archetypes, this
    function can be ignored — nothing else depends on it.

    Parameters
    ----------
    labels        : cluster label array (-1 = noise).
    archetype_map : dict mapping int -> str. Defaults to ARCHETYPE_MAP.

    Returns
    -------
    numpy array of archetype name strings, same length as labels.
    """
    mapping = archetype_map if archetype_map is not None else ARCHETYPE_MAP
    return np.array([mapping.get(int(lbl), f"Cluster {lbl}") for lbl in labels])
