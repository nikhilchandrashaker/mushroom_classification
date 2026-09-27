"""Shared utilities for the mushroom classification benchmark project.

Loaders, preprocessing pipelines, and evaluation helpers used across the
modeling, scaling, and explainability notebooks.
"""
import time

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix,
)
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RANDOM_STATE = 42

# --------------------------------------------------------------------------
# Loaders
# --------------------------------------------------------------------------

CLASSIC_COLUMNS = [
    "class", "cap-shape", "cap-surface", "cap-color", "bruises", "odor",
    "gill-attachment", "gill-spacing", "gill-size", "gill-color",
    "stalk-shape", "stalk-root", "stalk-surface-above-ring",
    "stalk-surface-below-ring", "stalk-color-above-ring",
    "stalk-color-below-ring", "veil-type", "veil-color", "ring-number",
    "ring-type", "spore-print-color", "population", "habitat",
]

CLASSIC_DECODE = {
    "class": {"e": "edible", "p": "poisonous"},
    "cap-shape": {"b": "bell", "c": "conical", "x": "convex", "f": "flat", "k": "knobbed", "s": "sunken"},
    "cap-surface": {"f": "fibrous", "g": "grooves", "y": "scaly", "s": "smooth"},
    "cap-color": {"n": "brown", "b": "buff", "c": "cinnamon", "g": "gray", "r": "green", "p": "pink",
                  "u": "purple", "e": "red", "w": "white", "y": "yellow"},
    "bruises": {"t": "bruises", "f": "no"},
    "odor": {"a": "almond", "l": "anise", "c": "creosote", "y": "fishy", "f": "foul",
             "m": "musty", "n": "none", "p": "pungent", "s": "spicy"},
    "gill-attachment": {"a": "attached", "d": "descending", "f": "free", "n": "notched"},
    "gill-spacing": {"c": "close", "w": "crowded", "d": "distant"},
    "gill-size": {"b": "broad", "n": "narrow"},
    "gill-color": {"k": "black", "n": "brown", "b": "buff", "h": "chocolate", "g": "gray",
                   "r": "green", "o": "orange", "p": "pink", "u": "purple", "e": "red",
                   "w": "white", "y": "yellow"},
    "stalk-shape": {"e": "enlarging", "t": "tapering"},
    "stalk-root": {"b": "bulbous", "c": "club", "u": "cup", "e": "equal", "z": "rhizomorphs",
                   "r": "rooted", "?": None},
    "stalk-surface-above-ring": {"f": "fibrous", "y": "scaly", "k": "silky", "s": "smooth"},
    "stalk-surface-below-ring": {"f": "fibrous", "y": "scaly", "k": "silky", "s": "smooth"},
    "stalk-color-above-ring": {"n": "brown", "b": "buff", "c": "cinnamon", "g": "gray", "o": "orange",
                                "p": "pink", "e": "red", "w": "white", "y": "yellow"},
    "stalk-color-below-ring": {"n": "brown", "b": "buff", "c": "cinnamon", "g": "gray", "o": "orange",
                                "p": "pink", "e": "red", "w": "white", "y": "yellow"},
    "veil-type": {"p": "partial", "u": "universal"},
    "veil-color": {"n": "brown", "o": "orange", "w": "white", "y": "yellow"},
    "ring-number": {"n": "none", "o": "one", "t": "two"},
    "ring-type": {"c": "cobwebby", "e": "evanescent", "f": "flaring", "l": "large",
                  "n": "none", "p": "pendant", "s": "sheathing", "z": "zone"},
    "spore-print-color": {"k": "black", "n": "brown", "b": "buff", "h": "chocolate", "r": "green",
                           "o": "orange", "u": "purple", "w": "white", "y": "yellow"},
    "population": {"a": "abundant", "c": "clustered", "n": "numerous", "s": "scattered",
                   "v": "several", "y": "solitary"},
    "habitat": {"g": "grasses", "l": "leaves", "m": "meadows", "p": "paths",
                "u": "urban", "w": "waste", "d": "woods"},
}


def load_classic(path="/mnt/user-data/uploads/agaricus-lepiota.data", decode=True):
    df = pd.read_csv(path, header=None, names=CLASSIC_COLUMNS, na_values="?")
    if decode:
        for col, mapping in CLASSIC_DECODE.items():
            df[col] = df[col].map(mapping)
    return df


def load_secondary(path="/mnt/user-data/uploads/secondary_data.csv"):
    df = pd.read_csv(path, sep=";")
    df["class"] = df["class"].map({"e": "edible", "p": "poisonous"})
    return df


# --------------------------------------------------------------------------
# Preprocessing
# --------------------------------------------------------------------------

def split_feature_types(df, target="class"):
    """Return (numeric_cols, categorical_cols) for a dataframe."""
    feature_cols = [c for c in df.columns if c not in (target, "family", "name")]
    numeric_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c])]
    categorical_cols = [c for c in feature_cols if c not in numeric_cols]
    return numeric_cols, categorical_cols


def make_onehot_preprocessor(numeric_cols, categorical_cols, scale=True):
    """Preprocessor for LR / DT / RF / XGBoost / LightGBM (dense one-hot)."""
    numeric_steps = [("impute", SimpleImputer(strategy="median"))]
    if scale:
        numeric_steps.append(("scale", StandardScaler()))
    numeric_pipe = Pipeline(numeric_steps)

    categorical_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value="missing")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    return ColumnTransformer([
        ("num", numeric_pipe, numeric_cols),
        ("cat", categorical_pipe, categorical_cols),
    ])


def prep_for_catboost(df, numeric_cols, categorical_cols):
    """CatBoost wants raw categorical strings (with an explicit 'missing' token)
    and can natively handle missing numeric values, but we fill for cleanliness."""
    X = df[numeric_cols + categorical_cols].copy()
    for c in categorical_cols:
        X[c] = X[c].astype("object").where(X[c].notna(), "missing").astype(str)
    cat_feature_idx = [X.columns.get_loc(c) for c in categorical_cols]
    return X, cat_feature_idx


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------

def get_scores(model, X):
    """Return positive-class ('poisonous') probability scores for ROC-AUC."""
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        return proba[:, 1]
    return model.decision_function(X)


def _manual_cv(estimator_factory, X, y_bin, cv_folds=5):
    """Cross-validation via a fresh-estimator factory instead of sklearn's clone().
    Needed for estimators (e.g. CatBoostClassifier with cat_features set) that
    sklearn's clone() cannot handle."""
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)
    accs, f1s, aucs = [], [], []
    is_df = hasattr(X, "iloc")
    for train_idx, test_idx in cv.split(X, y_bin):
        if is_df:
            X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        else:
            X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y_bin[train_idx], y_bin[test_idx]
        est = estimator_factory()
        est.fit(X_tr, y_tr)
        pred = est.predict(X_te)
        scores = get_scores(est, X_te)
        accs.append(accuracy_score(y_te, pred))
        f1s.append(f1_score(y_te, pred))
        aucs.append(roc_auc_score(y_te, scores))
    return {
        "cv_accuracy_mean": np.mean(accs), "cv_accuracy_std": np.std(accs),
        "cv_f1_mean": np.mean(f1s), "cv_roc_auc_mean": np.mean(aucs),
    }


def evaluate_model(name, estimator, X_train, X_test, y_train, y_test,
                    pos_label="poisonous", cv_folds=5, do_cv=True,
                    estimator_factory=None):
    """Fit, time, predict, and score a model. y_train/y_test are string labels.

    If `estimator_factory` is given (a zero-arg callable returning a fresh,
    unfitted estimator), it is used for cross-validation instead of
    sklearn's clone() — needed for models like CatBoostClassifier with
    cat_features set, which clone() cannot handle.
    """
    y_train_bin = (y_train == pos_label).astype(int).to_numpy()
    y_test_bin = (y_test == pos_label).astype(int).to_numpy()

    t0 = time.perf_counter()
    estimator.fit(X_train, y_train_bin)
    train_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    y_pred = estimator.predict(X_test)
    predict_time = time.perf_counter() - t0

    scores = get_scores(estimator, X_test)

    result = {
        "model": name,
        "accuracy": accuracy_score(y_test_bin, y_pred),
        "precision": precision_score(y_test_bin, y_pred),
        "recall": recall_score(y_test_bin, y_pred),
        "f1": f1_score(y_test_bin, y_pred),
        "roc_auc": roc_auc_score(y_test_bin, scores),
        "train_time_s": train_time,
        "predict_time_s": predict_time,
        "confusion_matrix": confusion_matrix(y_test_bin, y_pred),
    }

    if do_cv:
        if estimator_factory is not None:
            result.update(_manual_cv(estimator_factory, X_train, y_train_bin, cv_folds))
        else:
            cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)
            cv_res = cross_validate(
                estimator, X_train, y_train_bin, cv=cv,
                scoring=["accuracy", "f1", "roc_auc"], n_jobs=1,
            )
            result["cv_accuracy_mean"] = cv_res["test_accuracy"].mean()
            result["cv_accuracy_std"] = cv_res["test_accuracy"].std()
            result["cv_f1_mean"] = cv_res["test_f1"].mean()
            result["cv_roc_auc_mean"] = cv_res["test_roc_auc"].mean()

    return result, estimator


def results_to_frame(results):
    keep = ["model", "accuracy", "precision", "recall", "f1", "roc_auc",
            "cv_accuracy_mean", "cv_f1_mean", "cv_roc_auc_mean",
            "train_time_s", "predict_time_s"]
    rows = [{k: r.get(k) for k in keep} for r in results]
    return pd.DataFrame(rows).set_index("model")
