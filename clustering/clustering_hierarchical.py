"""
clustering/clustering_hierarchical.py
--------------------------------------
Hierarchical clustering for the F1 driver performance project.

Two methods are demonstrated following the lecture:
  - Ward's agglomerative clustering (bottom-up, minimises SSE increase)
  - Bisecting K-Means (divisive, top-down)

The number of clusters is determined by the CH score from K-Means tuning
(k=4), since Mojena's upper-tail rule produces too many clusters on small
datasets (n=77). This is noted as a design decision in the methodology.

Functions
---------
- fit_ward()           - fits Ward agglomerative clustering, returns linkage
                         matrix Z, labels, and cophenetic correlation
- fit_bisecting_kmeans() - fits BisectingKMeans with chosen k
- cluster_summary()    - mean feature values per cluster (for interpretation)
- assign_archetypes()  - maps cluster integers to readable name strings

Plot functions (call from report.ipynb)
---------------------------------------
- plot_dendrogram()    - dendrogram with driver names on x-axis
- plot_fusion_levels() - alpha (fusion level) vs number of clusters
"""

import numpy as np
import pandas as pd
from sklearn.cluster import BisectingKMeans
from sklearn.metrics import calinski_harabasz_score
from scipy.cluster.hierarchy import linkage, dendrogram, cut_tree, cophenet
from scipy.spatial.distance import pdist
import matplotlib.pyplot as plt
from clustering.utils import plot_clustering, plot_cluster_averages


# fill in after inspecting cluster_summary() — remove if not needed
ARCHETYPE_MAP: dict[int, str] = {
    0: "Cluster 0",
    1: "Cluster 1",
    2: "Cluster 2",
    3: "Cluster 3",
}


# Fit Ward agglomerative clustering
def fit_ward(
    X: pd.DataFrame,
    n_clusters: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Fit Ward's agglomerative clustering and cut the dendrogram at n_clusters.

    Ward's method merges clusters by minimising the increase in total
    within-cluster SSE — it tends to produce compact, roughly equal-sized
    clusters and is well suited to our PCA-transformed feature space.

    Note: Mojena's upper-tail rule (the lecture method for choosing the cut
    automatically) produces too many clusters on small datasets (n=77).
    We use n_clusters directly, anchored to k=4 from K-Means CH score tuning.

    Parameters
    ----------
    X          : feature DataFrame to cluster (PCA-transformed coordinates).
    n_clusters : number of clusters to cut the dendrogram into.

    Returns
    -------
    Z           : linkage matrix from scipy (shape n-1 x 4).
    labels      : cluster label array, one integer per driver.
    cophenetic  : cophenetic correlation coefficient — how well the dendrogram
                  preserves the original pairwise distances (higher is better).
    """
    Z = linkage(X, method="ward", optimal_ordering=True)
    labels = cut_tree(Z, n_clusters=n_clusters).flatten()
    cophenetic_corr = cophenet(Z, pdist(X))[0]
    print(f"Cophenetic correlation (Ward): {cophenetic_corr:.4f}")
    return Z, labels, cophenetic_corr


# Fit Bisecting K-Means (divisive)
def fit_bisecting_kmeans(
    X: pd.DataFrame,
    n_clusters: int,
    random_state: int = 0,
) -> tuple[BisectingKMeans, np.ndarray]:
    """
    Fit Bisecting K-Means divisive clustering with chosen n_clusters.

    BisectingKMeans repeatedly splits the largest cluster using K-Means
    (k=2) until n_clusters are reached. It is the divisive counterpart
    to Ward's agglomerative method.

    Parameters
    ----------
    X            : feature DataFrame to cluster.
    n_clusters   : number of clusters to produce.
    random_state : random seed for reproducibility.

    Returns
    -------
    model  : fitted BisectingKMeans object.
    labels : cluster label array.
    """
    model = BisectingKMeans(n_clusters=n_clusters, random_state=random_state)
    labels = model.fit_predict(X)
    return model, labels


# Mean feature values per cluster
def cluster_summary(
    features: pd.DataFrame,
    labels: np.ndarray,
) -> pd.DataFrame:
    """
    Mean of each feature per cluster, with overall mean as a reference row.

    Use this to interpret what each cluster represents before filling in
    ARCHETYPE_MAP. The 'n_drivers' column shows cluster size.

    Parameters
    ----------
    features : original (pre-PCA) feature DataFrame, one row per driver.
    labels   : cluster label array.

    Returns
    -------
    DataFrame — one row per cluster plus an 'Overall' row.
    """
    df = features.copy()
    df["cluster"] = labels
    summary = df.groupby("cluster").mean()
    summary["n_drivers"] = df.groupby("cluster").size()
    summary.index = [f"Cluster {i}" for i in summary.index]
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
    labels        : cluster label array.
    archetype_map : dict mapping int -> str. Defaults to ARCHETYPE_MAP.

    Returns
    -------
    numpy array of archetype name strings, same length as labels.
    """
    mapping = archetype_map if archetype_map is not None else ARCHETYPE_MAP
    return np.array([mapping.get(int(lbl), f"Cluster {lbl}") for lbl in labels])


# Plot dendrogram with driver name labels
def plot_dendrogram(Z: np.ndarray, meta: pd.DataFrame, ax=None):
    """
    Dendrogram with driver names on the x-axis.

    Parameters
    ----------
    Z    : linkage matrix from fit_ward.
    meta : metadata DataFrame with a 'driver_name' column, same order as X.
    ax   : optional matplotlib Axes.

    Returns
    -------
    fig, ax : matplotlib Figure and Axes.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(14, 5))
    else:
        fig = ax.get_figure()

    dend = dendrogram(Z, ax=ax, leaf_rotation=90, leaf_font_size=7)
    ax.set_xticklabels(
        [meta["driver_name"].iloc[int(t.get_text())] for t in ax.get_xticklabels()],
        rotation=90, ha="center", fontsize=7,
    )
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_ylabel("Distance", fontsize=11)
    ax.set_title("Ward Dendrogram — F1 Drivers", fontsize=13)
    plt.tight_layout()
    return fig, ax


# Plot fusion levels (alpha) vs number of clusters
def plot_fusion_levels(Z: np.ndarray, ax=None):
    """
    Plot the fusion level alpha at each merge step vs number of clusters.

    A large jump in alpha indicates a natural cut point in the hierarchy.
    Used to visually support the choice of n_clusters.

    Parameters
    ----------
    Z  : linkage matrix from fit_ward.
    ax : optional matplotlib Axes.

    Returns
    -------
    fig, ax : matplotlib Figure and Axes.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    else:
        fig = ax.get_figure()

    alpha = pd.Series(Z[:, 2]) - pd.Series(Z[:, 2]).shift(1).fillna(0)
    n_range = range(2, len(alpha) + 1)

    ax.plot(n_range, alpha[::-1].to_numpy(), "o-", lw=2, color="tab:blue")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xticks(list(n_range)[::2])
    ax.set_xlabel("Number of Clusters", fontsize=12)
    ax.set_ylabel(r"Fusion level $\alpha$", fontsize=12)
    ax.set_title("Fusion Levels — Ward Linkage", fontsize=13)
    return fig, ax
