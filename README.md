# Pitting Strategy to Podium: F1 Driver Archetype Clustering

**DMW 2301 – Data Mining and Wrangling | Mini Project 2**

## Project Overview

This project applies clustering analysis to the careers of 77 Formula 1 drivers (2000–2024)
to discover natural **driver performance archetypes**. We use K-Means, Ward's Hierarchical
Clustering, and DBSCAN on a 13-feature matrix engineered from race results, pit-stop records,
and qualifying data.

## Repository Structure

```
.
├── data/
│   └── raw/                     ← place all CSV files here
│       ├── circuits.csv
│       ├── constructor_results.csv
│       ├── constructor_standings.csv
│       ├── constructors.csv
│       ├── driver_standings.csv
│       ├── drivers.csv
│       ├── lap_times.csv
│       ├── pit_stops.csv
│       ├── qualifying.csv
│       ├── races.csv
│       ├── results.csv
│       ├── seasons.csv
│       ├── sprint_results.csv
│       └── status.csv
├── f1_clustering/               ← Python package
│   ├── __init__.py
│   ├── data.py                  ← data loading and merging
│   ├── features.py              ← feature engineering
│   ├── preprocessing.py         ← scaling + SVD pipeline
│   ├── clustering.py            ← K-Means, Ward's, DBSCAN
│   ├── evaluation.py            ← internal + external metrics
│   └── visualization.py         ← all plotting functions
├── tests/                       ← unit tests (pytest)
│   ├── __init__.py
│   ├── test_data.py
│   ├── test_features.py
│   ├── test_preprocessing.py
│   ├── test_clustering.py
│   └── test_evaluation.py
├── report.ipynb                 ← final analysis notebook
├── environment.yml
└── README.md
```

## Setup

```bash
# 1. Create and activate the conda environment
conda env create -f environment.yml
conda activate bsdsba2028-dmw-2301-mp2-lt-x

# 2. Place the Ergast CSV files in data/raw/

# 3. Run unit tests
pytest tests/ -v

# 4. Launch the report notebook
jupyter lab report.ipynb
```

## Running Tests

```bash
pytest tests/ -v --tb=short
```

All 50+ tests cover data loading, feature engineering, preprocessing, clustering
algorithms, and evaluation metrics.

## Data Source

Ergast Motor Racing Developer API — Formula 1 World Championship dataset (1950–2024).
Downloaded from: https://ergast.com/mrd/db/

## Authors

*(Fill in team member names and student IDs)*
