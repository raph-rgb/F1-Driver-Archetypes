"""
clustering/utils.py
-------------------
Shared plot helpers for all three clustering modules.

Functions
---------
- plot_clustering()        - scatter of points coloured by cluster label
- plot_cluster_averages()  - mean feature values per cluster vs overall mean
- plot_internal_metric()   - line + band chart for inertia / CH score / silhouette
- plot_neighbor_distances()- k-NN distances with z-score eps threshold (DBSCAN tuning)

All functions return (fig, ax) or (fig, ax, threshold) so they can be called
directly from report.ipynb. Adapted from prof's utils.py.
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms


# Scatter plot coloured by cluster label
def plot_clustering(data, column_1, column_2, labels, cluster_centers=None):
    """
    Scatter plot of data points coloured by cluster label.

    Parameters
    ----------
    data            : DataFrame with at least column_1 and column_2.
    column_1        : str, x-axis column name.
    column_2        : str, y-axis column name.
    labels          : array of cluster labels (integers, -1 = noise for DBSCAN).
    cluster_centers : optional DataFrame of centroids with same columns.

    Returns
    -------
    fig, ax : matplotlib Figure and Axes.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(data[column_1], data[column_2], c=labels, cmap="plasma",
               alpha=0.75, s=45)

    if cluster_centers is not None:
        ax.scatter(
            cluster_centers[column_1], cluster_centers[column_2],
            marker="*", ec="k", lw=1.5, s=180,
            c=np.unique(labels), cmap="plasma",
        )
        for j in range(len(cluster_centers)):
            ax.text(cluster_centers.loc[j, column_1],
                    cluster_centers.loc[j, column_2],
                    f" {j}", fontsize=9, fontweight="bold")

    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlabel(column_1, fontsize=12)
    ax.set_ylabel(column_2, fontsize=12)
    return fig, ax


# Mean feature values per cluster vs overall mean
def plot_cluster_averages(X, cluster_labels):
    """
    Line plot of mean feature values per cluster against the overall mean.

    Use this after clustering to interpret what each cluster represents.
    The black star line is the overall dataset average — cluster lines
    above or below it show where each group deviates.

    Parameters
    ----------
    X              : original feature DataFrame (pre-PCA), one row per driver.
    cluster_labels : array of cluster label integers.

    Returns
    -------
    fig, ax : matplotlib Figure and Axes.
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    X.mean(axis=0).plot(style="*", ax=ax, color="black",
                        markersize=10, linestyle="none", label="Overall mean")

    unique_labels = np.unique(cluster_labels)
    color_map = mpl.colormaps["plasma"].resampled(len(unique_labels)).colors
    for i in unique_labels:
        mask = cluster_labels == i
        label = "Noise" if i == -1 else f"Cluster {i}"
        color = "gray" if i == -1 else color_map[i]
        X.loc[mask].mean(axis=0).plot(
            style="o", ax=ax, color=color, linestyle="none", label=label)

    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="upper left", bbox_to_anchor=(1., 1.))
    ax.tick_params(axis="x", rotation=30)
    return fig, ax


# Internal metric plot (inertia / CH score / silhouette)
def plot_internal_metric(k_range, metric_mean, metric_std, metric_name,
                         ebar_scale=10, color="tab:blue"):
    """
    Line plot with shaded uncertainty band for any internal clustering metric.

    Parameters
    ----------
    k_range      : list of k values evaluated.
    metric_mean  : array of mean metric values per k.
    metric_std   : array of std of metric values per k.
    metric_name  : str label for the y-axis.
    ebar_scale   : multiplier for the std band width (default 10).
    color        : line and band colour.

    Returns
    -------
    fig, ax : matplotlib Figure and Axes.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.fill_between(k_range,
                    metric_mean + ebar_scale * metric_std,
                    metric_mean - ebar_scale * metric_std,
                    alpha=0.2, color=color)
    ax.plot(k_range, metric_mean, "o-", lw=2, ms=5, color=color)
    ax.set_xticks(list(k_range))
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlabel("Number of Clusters", fontsize=12)
    ax.set_ylabel(metric_name.title(), fontsize=12)
    return fig, ax


# k-NN distance plot for DBSCAN eps tuning
def plot_neighbor_distances(distances, num_std=3):
    """
    Sorted k-NN distance plot with a z-score upper threshold line.

    Used to determine eps for DBSCAN: set min_samples first, compute
    the tau-nearest neighbor distance for each point, then read off
    the threshold where distances spike (the tail of the distribution).

    Parameters
    ----------
    distances : 1-D array of tau-nearest neighbor distances.
    num_std   : number of standard deviations above the mean for threshold.

    Returns
    -------
    fig, ax   : matplotlib Figure and Axes.
    threshold : float, computed eps value (mean + num_std * std).
    """
    threshold = distances.mean() + num_std * distances.std()

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(range(len(distances)), sorted(distances), "o-", lw=2)
    ax.axhline(threshold, ls="--", lw=2, color="tab:orange")

    trans = transforms.blended_transform_factory(ax.transAxes, ax.transData)
    ylim_range = np.diff(ax.get_ylim())[0]
    ax.text(s=f"eps = {threshold:.3f}", x=0.02,
            y=threshold + ylim_range * 0.02,
            fontsize=11, color="tab:orange", weight="bold", transform=trans)

    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlabel("Data Index (sorted)", fontsize=12)
    ax.set_ylabel(r"$\tau$-neighbor distance", fontsize=12)
    return fig, ax, threshold
