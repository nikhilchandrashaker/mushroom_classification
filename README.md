# 🍄 Mushroom Edibility: Classic Classification to Large-Scale ML

Educational/ML benchmark project comparing the classic 1987 UCI mushroom dataset against a
larger, modern simulated dataset. **Not a foraging or mushroom-identification safety tool** —
see the caveat in `PROJECT_REPORT.md`.

## Contents

| File | Description |
|---|---|
| `PROJECT_REPORT.md` | Full write-up: findings from all 5 phases, results tables, conclusions |
| `mushroom_lib.py` | Shared loaders, preprocessing pipelines, and evaluation helpers |
| `mushroom_eda.ipynb` | Phase 1 — exploratory data analysis on both datasets |
| `mushroom_modeling.ipynb` | Phases 2 & 4 — model benchmarking + cross-dataset comparison |
| `mushroom_scaling.ipynb` | Phase 3 — learning-curve / data-scaling experiment |
| `mushroom_explainability.ipynb` | Phase 5 — decision-tree rules, feature importance, SHAP |
| `*.html` | Pre-rendered, already-executed versions of each notebook — open these to view results without running any code |
| `load_classic.py` | Standalone classic-dataset loader (superseded by `mushroom_lib.py`, kept for reference) |

## Data

Not included in this folder — place these alongside the files above before re-running:

- `agaricus-lepiota.data` — classic dataset (8,124 rows)
- `secondary_data.csv` — secondary dataset (61,069 rows, `;`-delimited)

Both notebooks and `mushroom_lib.py` default to loading them from
`/mnt/user-data/uploads/`; change the `path=` argument in `load_classic()` /
`load_secondary()` if running elsewhere.

## Setup

```bash
pip install pandas numpy scipy scikit-learn matplotlib seaborn \
            xgboost lightgbm catboost shap jupyter nbformat nbconvert
```

## Running

Each notebook is self-contained and only depends on `mushroom_lib.py` plus the two data
files above. Run them in any order — `mushroom_eda.ipynb` is the natural starting point.

```bash
jupyter nbconvert --to notebook --execute --inplace mushroom_eda.ipynb
# or open in Jupyter Lab / VS Code and run all cells
```

`mushroom_modeling.ipynb` takes a few minutes (Random Forest and CatBoost are the slowest
steps, especially on the 61k-row secondary dataset).

## Headline result

The classic dataset is nearly solved by its `odor` feature alone (~99–100% accuracy across
every model tested). The secondary dataset has no `odor` field and is a genuinely harder,
more realistic benchmark: Logistic Regression drops to ~86% accuracy there, while tree-based
and boosting models still reach ~99.9–100% — see `PROJECT_REPORT.md` for the full breakdown.
