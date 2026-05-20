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

# Constants
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


# Plot biplot (interactive Plotly)
def plot_biplot(
    X_pca: pd.DataFrame,
    pca: PCA,
    meta: pd.DataFrame,
    pc_x: int = 1,
    pc_y: int = 2,
):
    """
    Interactive Plotly biplot — driver scatter with feature loading arrows.

    Hover any dot to see the driver's name, nationality, and total races.
    Each arrow is a feature loading — direction shows correlation with PCs,
    length shows strength of influence. Arrows pointing the same way are
    correlated; opposite directions mean negatively correlated.

    Use in report.ipynb:
        fig = plot_biplot(results['X_pca'], results['pca'], results['meta'])
        fig.show()

    Parameters
    ----------
    X_pca  : transformed coordinate DataFrame (output of run_pca).
    pca    : fitted PCA object.
    meta   : metadata DataFrame with driver_name, nationality, total_races.
    pc_x   : PC on x-axis (1-indexed, default 1).
    pc_y   : PC on y-axis (1-indexed, default 2).

    Returns
    -------
    plotly Figure — call .show() in a notebook or .write_html() to save.
    """
    import plotly.graph_objects as go

    cx  = f"PC{pc_x}"
    cy  = f"PC{pc_y}"
    var_x = round(pca.explained_variance_ratio_[pc_x - 1] * 100, 1)
    var_y = round(pca.explained_variance_ratio_[pc_y - 1] * 100, 1)

    plot_df = meta.reset_index(drop=True).copy()
    plot_df["x"] = X_pca[cx].values
    plot_df["y"] = X_pca[cy].values

    # Arrow scaling — fit inside 35% of each axis range
    x_range = X_pca[cx].max() - X_pca[cx].min()
    y_range = X_pca[cy].max() - X_pca[cy].min()
    loadings_x = pca.components_[pc_x - 1]
    loadings_y = pca.components_[pc_y - 1]
    max_loading = max(np.abs(loadings_x).max(), np.abs(loadings_y).max())
    sx = (x_range * 0.35) / max_loading
    sy = (y_range * 0.35) / max_loading

    FEATURE_LABELS = {
        "avg_finish_position":   "Finish position",
        "avg_grid_position":     "Grid position",
        "avg_points_per_race":   "Points / race",
        "win_rate":              "Win rate",
        "podium_rate":           "Podium rate",
        "dnf_rate":              "DNF rate",
        "avg_fastest_lap_speed": "Fastest lap speed",
        "avg_best_quali_time":   "Best quali time",
    }

    fig = go.Figure()

    # Driver scatter
    fig.add_trace(go.Scatter(
        x=plot_df["x"],
        y=plot_df["y"],
        mode="markers",
        name="Drivers",
        marker=dict(color="#4C72B0", size=8, opacity=0.75,
                    line=dict(width=0.8, color="white")),
        customdata=np.stack([
            plot_df["driver_name"],
            plot_df["nationality"],
            plot_df["total_races"],
            plot_df["x"].round(2),
            plot_df["y"].round(2),
        ], axis=-1),
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>"
            "Nationality: %{customdata[1]}<br>"
            "Career races: %{customdata[2]}<br>"
            f"{cx}: %{{customdata[3]}}  |  {cy}: %{{customdata[4]}}"
            "<extra></extra>"
        ),
    ))

    # Loading arrows
    for i, feat in enumerate(FEATURE_COLS):
        ex = loadings_x[i] * sx
        ey = loadings_y[i] * sy
        label = FEATURE_LABELS.get(feat, feat)
        nudge = 0.08
        nx = ex + nudge * np.sign(ex + 1e-9)
        ny = ey + nudge * np.sign(ey + 1e-9)

        fig.add_annotation(
            x=ex, y=ey, ax=0, ay=0,
            axref="x", ayref="y", xref="x", yref="y",
            showarrow=True, arrowhead=3,
            arrowsize=1.2, arrowwidth=2,
            arrowcolor="#C44E52",
        )
        fig.add_annotation(
            x=nx, y=ny,
            text=f"<b>{label}</b>",
            showarrow=False,
            font=dict(size=10, color="#C44E52"),
            xref="x", yref="y",
            xanchor="center",
        )

    # Zero reference lines
    fig.add_shape(type="line",
        x0=X_pca[cx].min() - 0.5, x1=X_pca[cx].max() + 0.5, y0=0, y1=0,
        line=dict(color="rgba(100,100,100,0.3)", width=1, dash="dot"))
    fig.add_shape(type="line",
        x0=0, x1=0,
        y0=X_pca[cy].min() - 0.5, y1=X_pca[cy].max() + 0.5,
        line=dict(color="rgba(100,100,100,0.3)", width=1, dash="dot"))

    fig.update_layout(
        template="plotly_white",
        title=dict(
            text="<b>PCA Biplot — F1 Driver Performance (2000–2024)</b>",
            font=dict(size=16), x=0.5, xanchor="center",
        ),
        xaxis=dict(title=f"{cx} — {var_x}% variance explained",
                   showgrid=True, gridcolor="rgba(0,0,0,0.06)", zeroline=False),
        yaxis=dict(title=f"{cy} — {var_y}% variance explained",
                   showgrid=True, gridcolor="rgba(0,0,0,0.06)", zeroline=False),
        hoverlabel=dict(bgcolor="white", font_size=12),
        showlegend=False,
        width=820, height=600,
        margin=dict(l=60, r=40, t=60, b=60),
        plot_bgcolor="white",
        paper_bgcolor="white",
    )

    return fig


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
