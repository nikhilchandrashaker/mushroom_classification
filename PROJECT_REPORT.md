# 🍄 Mushroom Edibility: From Classic Classification to Large-Scale Machine Learning

**An ML/educational benchmark project — not a foraging or mushroom-identification safety tool.**
Both the classic dataset's own documentation and the newer secondary dataset's source material
are explicit that there is no simple rule for determining whether a real mushroom is safe to
eat. Nothing in this project should be used to make that decision.

---

## Project structure

| Phase | Notebook | Question |
|---|---|---|
| 1 — EDA | `mushroom_eda.ipynb` | What distinguishes edible vs. poisonous mushrooms in each dataset? |
| 2 — Benchmarking | `mushroom_modeling.ipynb` | Which algorithms perform well, and at what cost? |
| 3 — Scaling | `mushroom_scaling.ipynb` | How much data is enough? Do more features help? |
| 4 — Comparison | *(part of `mushroom_modeling.ipynb`)* | How do the two datasets compare head-to-head? |
| 5 — Explainability | `mushroom_explainability.ipynb` | Why does the model decide what it decides? |

All notebooks share one preprocessing/evaluation library, `mushroom_lib.py`, so results are
directly comparable across phases. Every notebook is self-contained and re-runnable (they only
depend on `mushroom_lib.py` and the original data files).

## The datasets

- **Classic** (`agaricus-lepiota.data`, UCI 1987): 8,124 mushrooms, 22 categorical features,
  23 species, class split 51.8% edible / 48.2% poisonous.
- **Secondary** (`secondary_data.csv`, 2020, simulated): 61,069 mushrooms, 20 features
  (17 categorical + 3 continuous: cap-diameter, stem-height, stem-width), simulated from
  173 real species profiles.

---

## Phase 1 — EDA highlights

- Both datasets are close to class-balanced (accuracy is a meaningful metric for both).
- The classic dataset has missing values in only one column (`stalk-root`, ~30%). The
  secondary dataset has much heavier structural missingness (up to ~95% missing for
  `veil-type`) — almost certainly reflecting genuinely absent anatomical structures rather
  than data-collection gaps.
- `odor` is an extremely strong, near-deterministic predictor in the classic dataset
  (Cramer's V ≈ 0.9+). Categories like *foul*, *fishy*, *creosote*, *musty* are essentially
  100% poisonous; *almond* and *anise* are 100% edible.
- The secondary dataset has **no `odor` field at all**, and its three numeric measurements
  overlap heavily between classes on their own — no single feature dominates the way `odor`
  does in the classic set.

## Phase 2 & 4 — Model benchmarking and cross-dataset comparison

Six models were trained on both datasets with an identical preprocessing pipeline (one-hot
encoding + median imputation for tree/linear/boosting models; native categorical handling for
CatBoost), evaluated with a held-out 20% test set plus 5-fold cross-validation.

| Dataset | Model | Accuracy | F1 | ROC-AUC |
|---|---|---|---|---|
| Classic | Logistic Regression | 0.9994 | 0.9994 | 1.0000 |
| Classic | Decision Tree | 1.0000 | 1.0000 | 1.0000 |
| Classic | Random Forest | 1.0000 | 1.0000 | 1.0000 |
| Classic | XGBoost | 1.0000 | 1.0000 | 1.0000 |
| Classic | LightGBM | 1.0000 | 1.0000 | 1.0000 |
| Classic | CatBoost | 1.0000 | 1.0000 | 1.0000 |
| Secondary | Logistic Regression | 0.8643 | 0.8779 | 0.9369 |
| Secondary | Decision Tree | 0.9992 | 0.9993 | 0.9992 |
| Secondary | Random Forest | 1.0000 | 1.0000 | 1.0000 |
| Secondary | XGBoost | 1.0000 | 1.0000 | 1.0000 |
| Secondary | LightGBM | 1.0000 | 1.0000 | 1.0000 |
| Secondary | CatBoost | 0.9996 | 0.9996 | 1.0000 |

**Key comparison takeaways:**
- Nearly every model reaches ~99–100% on the *classic* dataset, largely because `odor` makes
  it close to a solved problem.
- On the *secondary* dataset, Logistic Regression drops to ~86% — it can't capture the
  nonlinear feature interactions needed without `odor` — while every tree-based/boosting model
  still reaches 99.9–100%. This is strong evidence the **secondary dataset is the more
  realistic, harder benchmark** of the two.
- Training cost scales with dataset size as expected: Random Forest and CatBoost are markedly
  slower on the ~61k-row secondary dataset than on the ~8k-row classic one, while
  XGBoost/LightGBM stay comparatively fast at both scales.
- The Decision Tree trails the ensembles by only a fraction of a percentage point on both
  datasets, while being the only model whose full decision logic is directly readable —
  a real simplicity/performance trade-off worth surfacing to a stakeholder.

## Phase 3 — Data-scaling experiment

Using progressively larger training subsets of the secondary dataset (1,000 → 5,000 → 10,000
→ 25,000 → 50,000 → ~48,855 full training pool) against a single fixed validation set:

- Tree-based and boosting models (Decision Tree, Random Forest, LightGBM) reach near-maximal
  validation performance with only **1,000–5,000** rows — a small fraction of the full
  training pool.
- Logistic Regression improves more gradually as data grows, since it needs more examples to
  approximate a decision boundary the tree-based models capture natively from far less data.
- Training time grows roughly linearly (or worse, for Random Forest) with dataset size, while
  LightGBM stays comparatively cheap even at full scale.
- Restricting to the **top ~8 most important features** (by Random Forest importance) reaches
  validation performance close to using the full ~20-feature set, at a fraction of the encoded
  dimensionality — a useful option if a lighter production model were ever needed.

**Answer to "how much data is enough?":** for the nonlinear models, surprisingly little; more
data mainly helps the linear model close the gap.

## Phase 5 — Explainability

- A shallow (depth-4), fully human-readable Decision Tree reaches **~99% accuracy** on the
  classic dataset using rules built almost entirely from `odor` and `spore-print-color` —
  closely mirroring the hand-derived logical rules published in the original UCI
  documentation.
- The same shallow-tree approach on the secondary dataset reaches only **~71% accuracy** at
  the same depth — concrete, quantified evidence that this dataset genuinely lacks a single
  dominant feature and needs more, weaker signals combined to reach its ~99.9% full-tree
  ceiling.
- SHAP analysis of the LightGBM models confirms the same story: `odor` dominates the classic
  dataset's global SHAP importance almost completely, while the secondary dataset's importance
  is spread across `stem-width`, `stem-color`, `cap-diameter`, and several ring/spore-print
  categories.
- Individual SHAP waterfall plots give a template for a plain-language "why" answer for any
  specific mushroom, e.g. *"predicted poisonous mainly because of its odor category and spore
  print color"* (classic) vs. *"predicted poisonous mainly because of its stem width and stem
  color"* (secondary).

---

## Overall conclusions

1. **The two datasets are not directly comparable "apples to apples."** The classic dataset is
   close to a solved problem once `odor` is available; the secondary dataset — despite being
   7.5x larger — is a harder, more realistic benchmark precisely because it lacks that one
   dominant feature.
2. **Simple models are enough for the classic dataset; nonlinear models earn their keep on the
   secondary one.** This is visible in every phase: EDA (feature association), benchmarking
   (Logistic Regression's accuracy gap), scaling (learning-curve shape), and explainability
   (shallow-tree accuracy gap).
3. **Data efficiency, not data volume, is the story for tree-based models.** They plateau with
   a few thousand rows; only the linear model benefits substantially from the dataset's full
   scale.
4. **Boosting models (XGBoost/LightGBM) are the best all-around choice** across both datasets
   in this benchmark — matching or beating Random Forest's accuracy at a fraction of the
   training cost, especially as the dataset grows.

## Files in this project

- `mushroom_lib.py` — shared loaders, preprocessing, and evaluation helpers
- `load_classic.py` — standalone classic-dataset loader (superseded by `mushroom_lib.py`, kept
  for reference)
- `mushroom_eda.ipynb` / `.html` — Phase 1
- `mushroom_modeling.ipynb` / `.html` — Phases 2 & 4
- `mushroom_scaling.ipynb` / `.html` — Phase 3
- `mushroom_explainability.ipynb` / `.html` — Phase 5
- `PROJECT_REPORT.md` — this file
