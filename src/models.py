"""
models.py — Model training functions and hyperparameter grids.

Each function encapsulates the full training protocol for one model family:
  - Baseline  : fit on a stratified subsample with default hyperparameters.
  - Engineered: tune on a stratified subsample via RandomizedSearchCV (or
                GridSearchCV for LinearSVC), then retrain the winner on the
                full training set.

Separating training logic from the notebook achieves three goals:
  1. The notebook reads as a scientific narrative, not boilerplate.
  2. Hyperparameter grids are version-controlled in one place.
  3. The same functions can be called from scripts or a future CLI pipeline.

Public API — Baseline trainers
------------------------------
train_rf_baseline(X, y, seed, sample_size)  -> RandomForestClassifier
train_lgb_baseline(X, y, seed)              -> LGBMClassifier
train_svc_baseline(X, y, seed)              -> Pipeline(scaler + LinearSVC)

Public API — Tuned trainers
---------------------------
train_random_forest(X_train, y_train, X_tune, y_tune, cv, seed)
    -> (RandomForestClassifier, RandomizedSearchCV)

train_lightgbm(X_train, y_train, X_tune, y_tune, cv, seed)
    -> (LGBMClassifier, RandomizedSearchCV)

train_linear_svc(X_train, y_train, seed, sample_size)
    -> Pipeline(scaler + LinearSVC)

Hyperparameter grids (module-level constants)
---------------------------------------------
RF_PARAM_GRID, LGB_PARAM_GRID, SVC_PARAM_GRID
"""

import numpy as np
import lightgbm as lgb

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, StratifiedKFold

# ---------------------------------------------------------------------------
# Global default seed — override per call when needed for ablation studies
# ---------------------------------------------------------------------------
RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# Hyperparameter search grids
# Defined here so they appear in version control alongside the training logic,
# not buried inside a notebook cell.
# ---------------------------------------------------------------------------

RF_PARAM_GRID = {
    'n_estimators'     : [100, 200],
    'max_depth'        : [10, 15, 20],   # capped — unlimited depth overfits on 8M rows
    'min_samples_split': [20, 100],       # larger values reduce overfitting
    'max_features'     : ['sqrt', 'log2'],
}

LGB_PARAM_GRID = {
    'n_estimators'    : [100, 200],
    'num_leaves'      : [31, 63],         # controls tree complexity
    'learning_rate'   : [0.05, 0.1, 0.2],
    'feature_fraction': [0.7, 0.85, 1.0], # column subsampling per tree
    'reg_lambda'      : [1.0, 2.0],        # L2 regularisation
}

SVC_PARAM_GRID = {
    'svc__C'   : [0.01, 0.1, 1.0, 10.0],
    'svc__loss': ['hinge', 'squared_hinge'],
}

# ---------------------------------------------------------------------------
# Baseline trainers
# Fit on a stratified subsample with sensible defaults.
# Purpose: establish a performance floor before feature engineering.
# ---------------------------------------------------------------------------

def train_rf_baseline(X, y, seed=RANDOM_SEED):
    """
    Fit a RandomForestClassifier with default hyperparameters.

    Parameters
    ----------
    X, y : training features and labels (already subsampled by caller)
    seed : random seed for reproducibility

    Returns
    -------
    Fitted RandomForestClassifier.

    Design note: max_depth=15 prevents individual trees from memorising the
    training sample; ensemble averaging handles the rest.
    """
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        random_state=seed,
        n_jobs=-1,
    )
    model.fit(X, y)
    return model


def train_lgb_baseline(X, y, seed=RANDOM_SEED):
    """
    Fit a LGBMClassifier with default hyperparameters.

    Parameters
    ----------
    X, y : training features and labels (already subsampled by caller)
    seed : random seed for reproducibility

    Returns
    -------
    Fitted LGBMClassifier.
    """
    model = lgb.LGBMClassifier(
        objective='multiclass',
        num_class=3,
        n_estimators=100,
        learning_rate=0.1,
        num_leaves=31,
        feature_fraction=0.8,
        bagging_fraction=0.8,
        bagging_freq=1,
        n_jobs=-1,
        random_state=seed,
        verbose=-1,
    )
    model.fit(X, y)
    return model


def train_svc_baseline(X, y, seed=RANDOM_SEED):
    """
    Fit a LinearSVC baseline wrapped in a StandardScaler pipeline.

    LinearSVC is sensitive to feature scale, so StandardScaler is always
    applied as part of the pipeline.  This ensures that the scaler is fitted
    only on training data and applied consistently to validation/test data.

    Parameters
    ----------
    X, y : training features and labels (already subsampled by caller)
    seed : random seed for reproducibility

    Returns
    -------
    Fitted Pipeline(StandardScaler + LinearSVC).
    """
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('svc', LinearSVC(C=1.0, max_iter=10_000, random_state=seed)),
    ])
    model.fit(X, y)
    return model


# ---------------------------------------------------------------------------
# Tuned trainers
# Two-stage: hyperparameter search on a tuning sample → full retrain on X_train.
# ---------------------------------------------------------------------------

def train_random_forest(X_train, y_train, X_tune, y_tune, cv,
                        seed=RANDOM_SEED, n_iter=8):
    """
    Tune and train a RandomForestClassifier.

    Stage 1 — RandomizedSearchCV on X_tune/y_tune (typically 500K rows,
              5-fold stratified CV, scored by macro F1).
    Stage 2 — Retrain the best configuration on the full X_train (8.1M rows).

    The two-stage approach keeps hyperparameter search tractable (~minutes)
    while ensuring the final model benefits from the full training data.

    Parameters
    ----------
    X_train, y_train : full training split (~8.1M rows)
    X_tune, y_tune   : stratified tuning subsample (typically 500K rows)
    cv               : pre-built StratifiedKFold split list (shared across models
                       for fair comparison — same folds, same data)
    seed             : random seed
    n_iter           : number of hyperparameter combinations to sample

    Returns
    -------
    (best_model, search_object)
        best_model   : RandomForestClassifier fitted on X_train
        search_object: RandomizedSearchCV, gives access to cv_results_ for
                       statistical significance testing (see Section 3.4)
    """
    search = RandomizedSearchCV(
        RandomForestClassifier(random_state=seed, n_jobs=-1),
        RF_PARAM_GRID,
        n_iter=n_iter,
        cv=cv,
        scoring='f1_macro',
        n_jobs=1,          # outer loop uses 1 job; inner RF uses n_jobs=-1
        random_state=seed,
        verbose=1,
    )
    search.fit(X_tune, y_tune)
    print(f'RF — best params : {search.best_params_}')
    print(f'RF — best CV F1  : {search.best_score_:.4f}')

    best_model = RandomForestClassifier(
        **search.best_params_,
        random_state=seed,
        n_jobs=-1,
    )
    best_model.fit(X_train, y_train)
    return best_model, search


def train_lightgbm(X_train, y_train, X_tune, y_tune, cv,
                   seed=RANDOM_SEED, n_iter=8):
    """
    Tune and train a LGBMClassifier.

    Same two-stage protocol as train_random_forest.  LightGBM's gradient-
    boosting strategy differs fundamentally from RF (sequential vs. parallel
    trees), so it is kept as a separate function with its own param grid.

    Parameters
    ----------
    (same as train_random_forest)

    Returns
    -------
    (best_model, search_object)
        best_model   : LGBMClassifier fitted on X_train
        search_object: RandomizedSearchCV object
    """
    base_lgb = lgb.LGBMClassifier(
        objective='multiclass',
        num_class=3,
        bagging_fraction=0.8,
        bagging_freq=1,
        n_jobs=-1,
        random_state=seed,
        verbose=-1,
    )
    search = RandomizedSearchCV(
        base_lgb,
        LGB_PARAM_GRID,
        n_iter=n_iter,
        cv=cv,
        scoring='f1_macro',
        random_state=seed,
        verbose=1,
    )
    search.fit(X_tune, y_tune)
    print(f'LGB — best params : {search.best_params_}')
    print(f'LGB — best CV F1  : {search.best_score_:.4f}')

    best_model = lgb.LGBMClassifier(
        **search.best_params_,
        objective='multiclass',
        num_class=3,
        bagging_fraction=0.8,
        bagging_freq=1,
        n_jobs=-1,
        random_state=seed,
        verbose=-1,
    )
    best_model.fit(X_train, y_train)
    return best_model, search


def train_linear_svc(X_train, y_train, seed=RANDOM_SEED, sample_size=200_000):
    """
    Tune and train a LinearSVC (wrapped in a StandardScaler pipeline).

    LinearSVC uses the liblinear solver, which is CPU-bound and scales linearly
    with sample size.  A 200K stratified subsample is used for the grid search;
    the best estimator is then refitted on the full training set.

    Grid search uses GridSearchCV (not Randomized) because the SVC grid is small
    (4 × 2 = 8 combinations) and exhaustive search is affordable.

    Parameters
    ----------
    X_train, y_train : full training split
    seed             : random seed
    sample_size      : rows used for hyperparameter search (default 200K)

    Returns
    -------
    Fitted Pipeline(StandardScaler + LinearSVC) on full X_train.
    """
    rng = np.random.default_rng(seed)
    n   = min(sample_size, len(X_train))
    idx = rng.integers(0, len(X_train), size=n)

    X_s = X_train.iloc[idx].reset_index(drop=True)
    y_s = y_train.iloc[idx].reset_index(drop=True)

    cv_svc = list(
        StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
        .split(X_s, y_s)
    )

    search = GridSearchCV(
        Pipeline([
            ('scaler', StandardScaler()),
            ('svc', LinearSVC(max_iter=10_000, random_state=seed)),
        ]),
        SVC_PARAM_GRID,
        cv=cv_svc,
        scoring='f1_macro',
        n_jobs=-1,
        verbose=1,
    )
    search.fit(X_s, y_s)
    print(f'SVC — best params : {search.best_params_}')
    print(f'SVC — best CV F1  : {search.best_score_:.4f}')

    best_model = search.best_estimator_
    best_model.fit(X_train, y_train)
    return best_model
