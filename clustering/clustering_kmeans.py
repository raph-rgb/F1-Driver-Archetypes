"""
clustering/clustering_kmeans.py
--------------------------------
K-Means clustering for the F1 driver performance project.

Functions
---------
- tune_kmeans()       - sweeps k, records inertia + silhouette + CH score
                        across multiple random trials
- fit_kmeans()        - fits final model using best-inertia seed
- cluster_summary()   - mean feature values per cluster (for interpretation)
- assign_archetypes() - maps cluster integers to readable name strings
                        (fill in ARCHETYPE_MAP after inspecting cluster_summary)

Plot functions (call from report.ipynb)
---------------------------------------
- plot_elbow()        - inertia vs k with uncertainty band
- plot_ch_score()     - CH score vs k with uncertainty band
- plot_silhouette()   - silhouette score vs k with uncertainty band
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabasz_score, silhouette_score
from clustering.utils import plot_clustering, plot_cluster_averages, plot_internal_metric


# Constants
NUM_TRIALS = 20
K_RANGE    = range(2, 11)

# fill in after inspecting cluster_summary() — remove if not needed
ARCHETYPE_MAP: dict[int, str] = {
    0: "Cluster 0",
    1: "Cluster 1",
    2: "Cluster 2",
    3: "Cluster 3",
}


# Sweep k to find optimal number of clusters
def tune_kmeans(
    X: pd.DataFrame,
    k_range: range = K_RANGE,
    num_trials: int = NUM_TRIALS,
) -> dict:
    """
    Run KMeans across k_range with num_trials random seeds per k.

    Records inertia, silhouette score, and CH score per trial then averages.
    Inertia starts at k=1; silhouette and CH start at k=2.

    Parameters
    ----------
    X          : feature DataFrame to cluster (PCA-transformed coordinates).
    k_range    : range of k values to try.
    num_trials : number of random seeds per k.

    Returns
    -------
    dict with keys:
        'k_range'         - list of all k values
        'inertia_mean'    - mean inertia per k (numpy array)
        'inertia_std'     - std of inertia per k (numpy array)
        'sil_mean'        - mean silhouette per k, k>=2 (numpy array)
        'sil_std'         - std of silhouette per k, k>=2 (numpy array)
        'ch_mean'         - mean CH score per k, k>=2 (numpy array)
        'ch_std'          - std of CH score per k, k>=2 (numpy array)
        'multi_k_range'   - k values for sil and CH (starts at 2)
    """
    inertia_mean, inertia_std = [], []
    sil_mean, sil_std = [], []
    ch_mean, ch_std = [], []
    multi_k_range = [k for k in k_range if k >= 2]

    for k in k_range:
        inertias, sils, chs = [], [], []
        for seed in range(num_trials):
            km = KMeans(n_clusters=k, random_state=seed, n_init="auto")
            labels = km.fit_predict(X)
            inertias.append(km.inertia_)
            if k >= 2:
                sils.append(silhouette_score(X, labels))
                chs.append(calinski_harabasz_score(X, labels))
        inertia_mean.append(np.mean(inertias))
        inertia_std.append(np.std(inertias))
        if k >= 2:
            sil_mean.append(np.mean(sils))
            sil_std.append(np.std(sils))
            ch_mean.append(np.mean(chs))
            ch_std.append(np.std(chs))

    return {
        "k_range":       list(k_range),
        "inertia_mean":  np.array(inertia_mean),
        "inertia_std":   np.array(inertia_std),
        "sil_mean":      np.array(sil_mean),
        "sil_std":       np.array(sil_std),
        "ch_mean":       np.array(ch_mean),
        "ch_std":        np.array(ch_std),
        "multi_k_range": multi_k_range,
    }


# Fit final KMeans with best seed
def fit_kmeans(
    X: pd.DataFrame,
    k: int,
    num_trials: int = NUM_TRIALS,
) -> tuple[KMeans, np.ndarray, pd.DataFrame]:
    """
    Fit final KMeans with chosen k, selecting the best seed by lowest inertia.

    Parameters
    ----------
    X          : feature DataFrame to cluster.
    k          : chosen number of clusters.
    num_trials : number of seeds to try.

    Returns
    -------
    kmeans          : fitted KMeans object.
    labels          : cluster label array.
    cluster_centers : DataFrame of centroids, columns match X.
    """
    inertias = []
    for seed in range(num_trials):
        km = KMeans(n_clusters=k, random_state=seed, n_init="auto")
        km.fit(X)
        inertias.append(km.inertia_)

    best = int(np.argmin(inertias))
    kmeans = KMeans(n_clusters=k, random_state=best, n_init="auto")
    labels = kmeans.fit_predict(X)
    centers = pd.DataFrame(kmeans.cluster_centers_, columns=X.columns)

    print(f"Best seed: {best}  |  Inertia: {kmeans.inertia_:.4f}")
    return kmeans, labels, centers


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
    labels   : cluster label array from fit_kmeans.

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


# Plot inertia elbow curve
def plot_elbow(tune_results: dict, ax=None):
    """
    Inertia vs k with shaded uncertainty band.

    The elbow — where inertia stops dropping sharply — suggests a good k.

    Parameters
    ----------
    tune_results : output dict from tune_kmeans.
    ax           : optional matplotlib Axes.

    Returns
    -------
    fig, ax : matplotlib Figure and Axes.
    """
    return plot_internal_metric(
        tune_results["k_range"],
        tune_results["inertia_mean"],
        tune_results["inertia_std"],
        "Inertia",
        ebar_scale=1,
        color="tab:blue",
    )


# Plot CH score curve
def plot_ch_score(tune_results: dict, ax=None):
    """
    Calinski-Harabasz score vs k with shaded uncertainty band.

    Higher = better defined clusters. Peak value indicates optimal k.

    Parameters
    ----------
    tune_results : output dict from tune_kmeans.
    ax           : optional matplotlib Axes.

    Returns
    -------
    fig, ax : matplotlib Figure and Axes.
    """
    return plot_internal_metric(
        tune_results["multi_k_range"],
        tune_results["ch_mean"],
        tune_results["ch_std"],
        "Calinski-Harabasz Score",
        ebar_scale=1,
        color="tab:orange",
    )


# Plot silhouette score curve
def plot_silhouette(tune_results: dict, ax=None):
    """
    Silhouette score vs k with shaded uncertainty band.

    Higher = better separation between clusters. Ranges from -1 to 1.

    Parameters
    ----------
    tune_results : output dict from tune_kmeans.
    ax           : optional matplotlib Axes.

    Returns
    -------
    fig, ax : matplotlib Figure and Axes.
    """
    return plot_internal_metric(
        tune_results["multi_k_range"],
        tune_results["sil_mean"],
        tune_results["sil_std"],
        "Silhouette Score",
        ebar_scale=1,
        color="tab:green",
    )
