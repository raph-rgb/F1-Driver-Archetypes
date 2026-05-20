"""
pca.py
------
Handles all preprocessing and PCA steps for the F1 driver clustering project.

Pipeline
--------
1. load_features()         - loads data/combined/data.csv, separates metadata
                             from numeric features
2. impute_features()       - fills the 6 null avg_fastest_lap_speed values
                             with the column median
3. scale_features()        - standardizes all features to mean=0, std=1
                             (required before PCA so no feature dominates
                             just because of its scale)
4. run_pca()               - fits PCA on the scaled matrix, returns the
                             fitted PCA object and the transformed coordinates
5. choose_n_components()   - finds how many PCs are needed to hit a target
                             cumulative explained variance (default 90%)
6. get_loadings()          - returns the PC loadings table (eigenvectors)
                             for interpretation

Plot functions (called from report.ipynb)
---------------------------------------------
- plot_scree()             - bar + line chart of explained variance per PC
- plot_cumulative_variance() - cumulative explained variance with threshold line
- plot_biplot()            - 2D scatter of drivers in PC space with
                             feature loading arrows overlaid
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

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

META_COLS = ["driverId", "driver_name", "nationality", "total_races"]

VARIANCE_THRESHOLD = 0.90   # keep PCs that together explain at least this much


# Loading features
def load_features(path: str = "data/combined/data.csv") -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load the combined dataset and split into metadata and numeric features.

    Parameters
    ----------
    path : str
        Path to data/combined/data.csv.

    Returns
    -------
    meta : DataFrame
        Columns: driverId, driver_name, nationality, total_races.
        Used for labelling plots and interpreting clusters — not fed into PCA.
    features : DataFrame
        The 8 numeric feature columns only.
    """
    df = pd.read_csv(path)
    return df[META_COLS].copy(), df[FEATURE_COLS].copy()


# 2. Impute missing values

def impute_features(features: pd.DataFrame) -> pd.DataFrame:
    """
    Fill missing values using column-wise median imputation.

    The only column with nulls is avg_fastest_lap_speed (6 drivers whose
    entire careers predate 2004, when the stat began being recorded).
    Median imputation is preferred over mean here because it is robust
    to the skew introduced by elite drivers with unusually high speeds.

    Parameters
    ----------
    features : DataFrame of raw numeric features (may contain NaNs).

    Returns
    -------
    DataFrame of the same shape with no NaN values.
    """
    imputer = SimpleImputer(strategy="median")
    imputed = imputer.fit_transform(features)
    return pd.DataFrame(imputed, columns=features.columns, index=features.index)


# Perform standard scaling
def scale_features(features: pd.DataFrame) -> tuple[pd.DataFrame, StandardScaler]:
    """
    Standardize all features to mean=0, standard deviation=1.

    This is mandatory before PCA. Without scaling, features with larger
    numeric ranges (e.g. avg_fastest_lap_speed in the 190-215 range)
    would dominate the principal components purely because of their scale,
    not because they are more informative.

    Parameters
    ----------
    features : imputed DataFrame of numeric features.

    Returns
    -------
    scaled   : DataFrame of the same shape, standardized.
    scaler   : the fitted StandardScaler (kept so you can inverse-transform
               later if needed).
    """
    scaler = StandardScaler()
    scaled_array = scaler.fit_transform(features)
    scaled = pd.DataFrame(scaled_array, columns=features.columns, index=features.index)
    return scaled, scaler


# Perform Principal Component Analysis
def run_pca(scaled: pd.DataFrame) -> tuple[PCA, pd.DataFrame]:
    """
    Fit PCA on the scaled feature matrix and return all components.

    We fit with all n_components first so we can inspect the full
    explained variance curve before deciding how many to keep.

    Parameters
    ----------
    scaled : standardized feature DataFrame.

    Returns
    -------
    pca          : fitted sklearn PCA object.
                   Key attributes:
                     .explained_variance_ratio_  - fraction of variance per PC
                     .components_                - eigenvectors (loadings)
    X_pca        : DataFrame of transformed coordinates, one row per driver,
                   columns named PC1, PC2, ... PCn.
    """
    pca = PCA()
    transformed = pca.fit_transform(scaled)

    pc_cols = [f"PC{i+1}" for i in range(transformed.shape[1])]
    X_pca = pd.DataFrame(transformed, columns=pc_cols, index=scaled.index)

    return pca, X_pca



# Determine number of components
def choose_n_components(
    pca: PCA,
    threshold: float = VARIANCE_THRESHOLD,
) -> int:
    """
    Return the minimum number of PCs needed to reach the variance threshold.

    Follows the convention from lecture: set a cumulative explained variance
    target (default 90%) and keep the smallest number of PCs that meets it.

    Parameters
    ----------
    pca       : fitted PCA object.
    threshold : float between 0 and 1, target cumulative explained variance.

    Returns
    -------
    n : int, number of components to retain.
    """
    cumulative = np.cumsum(pca.explained_variance_ratio_)
    n = int(np.searchsorted(cumulative, threshold) + 1)
    print(
        f"{n} component(s) needed to explain "
        f"{cumulative[n-1]*100:.1f}% of variance "
        f"(threshold: {threshold*100:.0f}%)"
    )
    return n



# Creating the loadings table
def get_loadings(pca: PCA, n_components: int) -> pd.DataFrame:
    """
    Return the PCA loadings table for the first n_components PCs.

    Each row is a principal component; each column is an original feature.
    Loadings tell you how much each original feature contributes to each PC.
    Large absolute values mean strong influence, while sign indicates direction.

    Parameters
    ----------
    pca          : fitted PCA object.
    n_components : how many PCs to include.

    Returns
    -------
    DataFrame of shape (n_components, n_features), indexed PC1..PCn.
    """
    return pd.DataFrame(
        pca.components_[:n_components],
        columns=FEATURE_COLS,
        index=[f"PC{i+1}" for i in range(n_components)],
    )


# Plot scree plot
def plot_scree(pca: PCA, ax=None) -> plt.Axes:
    """
    Bar chart of individual explained variance ratio per component,
    with a line connecting the tops (scree plot).

    Use this to visually confirm where variance drops off sharply.

    Parameters
    ----------
    pca : fitted PCA object.
    ax  : optional matplotlib Axes to draw on. Creates one if not provided.

    Returns
    -------
    ax : matplotlib Axes.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))

    ratios = pca.explained_variance_ratio_
    pcs = [f"PC{i+1}" for i in range(len(ratios))]

    ax.bar(pcs, ratios * 100, color="#4C72B0", alpha=0.85, zorder=2)
    ax.plot(pcs, ratios * 100, color="#C44E52", marker="o", linewidth=1.5, zorder=3)

    ax.set_xlabel("Principal Component")
    ax.set_ylabel("Explained Variance (%)")
    ax.set_title("Scree Plot — Variance Explained per PC")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", linestyle="--", alpha=0.4, zorder=1)

    return ax



# Plot cumulative explained variance
def plot_cumulative_variance(
    pca: PCA,
    threshold: float = VARIANCE_THRESHOLD,
    ax=None,
) -> plt.Axes:
    """
    Line plot of cumulative explained variance with a horizontal threshold line.

    Parameters
    ----------
    pca       : fitted PCA object.
    threshold : float, the variance threshold line to draw (default 0.90).
    ax        : optional matplotlib Axes.

    Returns
    -------
    ax : matplotlib Axes.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))

    cumvar = np.cumsum(pca.explained_variance_ratio_) * 100
    pcs = range(1, len(cumvar) + 1)

    ax.plot(pcs, cumvar, marker="o", color="#4C72B0", linewidth=2)
    ax.axhline(
        threshold * 100,
        color="#C44E52",
        linestyle="--",
        linewidth=1.5,
        label=f"{threshold*100:.0f}% threshold",
    )

    # Mark the crossing point
    n = choose_n_components(pca, threshold)
    ax.axvline(n, color="gray", linestyle=":", linewidth=1.2)
    ax.annotate(
        f"PC{n}\n({cumvar[n-1]:.1f}%)",
        xy=(n, cumvar[n - 1]),
        xytext=(n + 0.3, cumvar[n - 1] - 8),
        fontsize=9,
        color="gray",
    )

    ax.set_xlabel("Number of Components")
    ax.set_ylabel("Cumulative Explained Variance (%)")
    ax.set_title("Cumulative Explained Variance vs Number of PCs")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(linestyle="--", alpha=0.4)

    return ax


# Plot biplot
def plot_biplot(
    X_pca: pd.DataFrame,
    pca: PCA,
    meta: pd.DataFrame,
    pc_x: int = 1,
    pc_y: int = 2,
    ax=None,
) -> plt.Axes:
    """
    Biplot: driver positions in PC space with feature loading arrows overlaid.

    The scatter shows where each driver lands after PCA transformation.
    The arrows show how strongly and in what direction each original feature
    pulls — arrows pointing the same way are correlated; opposite = negatively
    correlated. Driver names are annotated on hover (static labels would
    overlap at 77 points, so we label a few standout drivers only).

    Parameters
    ----------
    X_pca  : transformed coordinate DataFrame (output of run_pca).
    pca    : fitted PCA object (for loadings and variance ratios).
    meta   : metadata DataFrame (for driver names).
    pc_x   : which PC to put on the x-axis (1-indexed, default 1).
    pc_y   : which PC to put on the y-axis (1-indexed, default 2).
    ax     : optional matplotlib Axes.

    Returns
    -------
    ax : matplotlib Axes.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 8))

    cx = f"PC{pc_x}"
    cy = f"PC{pc_y}"

    # Scatter drivers
    ax.scatter(
        X_pca[cx], X_pca[cy],
        color="#4C72B0", alpha=0.6, s=40, zorder=3,
    )

    # Annotate a few standout drivers for interpretation
    highlight = [
        "Lewis Hamilton", "Michael Schumacher", "Max Verstappen",
        "Fernando Alonso", "Kimi Räikkönen", "Sebastian Vettel",
        "Romain Grosjean", "Pastor Maldonado",
    ]
    meta_reset = meta.reset_index(drop=True)
    for idx, row in meta_reset.iterrows():
        if row["driver_name"] in highlight:
            ax.annotate(
                row["driver_name"],
                (X_pca[cx].iloc[idx], X_pca[cy].iloc[idx]),
                fontsize=7.5,
                xytext=(4, 4),
                textcoords="offset points",
                path_effects=[pe.withStroke(linewidth=2, foreground="white")],
            )

    # Loading arrows (scale to fit the scatter)
    scale = (X_pca[cx].abs().max() + X_pca[cy].abs().max()) / 2
    loadings_x = pca.components_[pc_x - 1]
    loadings_y = pca.components_[pc_y - 1]

    for i, feat in enumerate(FEATURE_COLS):
        ax.annotate(
            "",
            xy=(loadings_x[i] * scale, loadings_y[i] * scale),
            xytext=(0, 0),
            arrowprops=dict(arrowstyle="->", color="#C44E52", lw=1.5),
            zorder=4,
        )
        ax.text(
            loadings_x[i] * scale * 1.12,
            loadings_y[i] * scale * 1.12,
            feat.replace("avg_", "").replace("_", " "),
            fontsize=8,
            color="#C44E52",
            ha="center",
        )

    var_x = pca.explained_variance_ratio_[pc_x - 1] * 100
    var_y = pca.explained_variance_ratio_[pc_y - 1] * 100

    ax.set_xlabel(f"{cx} ({var_x:.1f}% variance)")
    ax.set_ylabel(f"{cy} ({var_y:.1f}% variance)")
    ax.set_title(f"PCA Biplot — {cx} vs {cy}")
    ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    ax.axvline(0, color="gray", linewidth=0.5, linestyle="--")
    ax.spines[["top", "right"]].set_visible(False)

    return ax


# Full pipeline
def run_pca_pipeline(
    path: str = "data/combined/data.csv",
    threshold: float = VARIANCE_THRESHOLD,
) -> dict:
    """
    End-to-end PCA pipeline: load -> impute -> scale -> PCA -> choose n.

    Parameters
    ----------
    path      : path to data/combined/data.csv
    threshold : cumulative variance threshold for choosing n_components

    Returns
    -------
    dict with keys:
        'meta'         - metadata DataFrame (driver names etc.)
        'features'     - raw feature DataFrame (pre-scaling)
        'features_imp' - imputed feature DataFrame
        'X_scaled'     - standardized feature DataFrame
        'scaler'       - fitted StandardScaler
        'pca'          - fitted PCA object
        'X_pca'        - transformed coordinates (all components)
        'n_components' - chosen number of components at threshold
        'loadings'     - loadings DataFrame for chosen components
    """
    meta, features      = load_features(path)
    features_imp        = impute_features(features)
    X_scaled, scaler    = scale_features(features_imp)
    pca, X_pca          = run_pca(X_scaled)
    n                   = choose_n_components(pca, threshold)
    loadings            = get_loadings(pca, n)

    return {
        "meta":          meta,
        "features":      features,
        "features_imp":  features_imp,
        "X_scaled":      X_scaled,
        "scaler":        scaler,
        "pca":           pca,
        "X_pca":         X_pca,
        "n_components":  n,
        "loadings":      loadings,
    }


# Run script
if __name__ == "__main__":
    results = run_pca_pipeline()

    print("\nExplained variance per PC:")
    for i, v in enumerate(results["pca"].explained_variance_ratio_):
        cum = np.cumsum(results["pca"].explained_variance_ratio_)[i]
        print(f"  PC{i+1}: {v*100:.2f}%  (cumulative: {cum*100:.2f}%)")

    print(f"\nRetaining {results['n_components']} components at 90% threshold")

    print("\nLoadings table:")
    print(results["loadings"].to_string())

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    plot_scree(results["pca"], ax=axes[0])
    plot_cumulative_variance(results["pca"], ax=axes[1])
    plot_biplot(results["X_pca"], results["pca"], results["meta"], ax=axes[2])
    plt.tight_layout()
    plt.savefig("data/combined/pca_overview.png", dpi=150)
    print("\nSaved pca_overview.png to data/combined/")
