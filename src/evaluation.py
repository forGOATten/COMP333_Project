"""
evaluation.py — Unified model evaluation protocol for the Bixi-Weather project.

All models (regression and classification) are scored through the functions
defined here.  Keeping evaluation logic in one place guarantees that every
model comparison in the notebook uses identical metric definitions and avoids
subtle inconsistencies (e.g., different average= arguments to f1_score).

Public API
----------
Constants
    METRICS_REG  : list[str]   Regression metric labels
    METRICS      : list[str]   Classification metric labels
    CLASS_NAMES  : list[str]   Human-readable duration class labels

Functions
    evaluate_reg(model, X, y)          -> list[float]
    evaluate_model(model, X, y)        -> list[float]
    bootstrap_ci(model, X, y, ...)     -> (mean, lo, hi)
"""

import numpy as np
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, f1_score, roc_auc_score,
)

# ---------------------------------------------------------------------------
# Metric label registries
# Used as DataFrame indices throughout the notebook so tables are self-labelled.
# ---------------------------------------------------------------------------
METRICS_REG  = ['RMSE (Minutes)', 'MAE (Minutes)', 'R\u00b2 Score']
METRICS      = ['Accuracy', 'Macro F1', 'ROC-AUC (OVR)']
CLASS_NAMES  = ['Short', 'Medium', 'Long']   # overwritten by data-driven labels


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _softmax(Z):
    """
    Row-wise softmax: converts LinearSVC decision scores to pseudo-probabilities.

    LinearSVC does not expose predict_proba().  Softmax-normalising the raw
    decision function output gives values that sum to 1 per sample and can be
    passed to roc_auc_score(..., multi_class='ovr').

    Note: these are not calibrated probabilities — they preserve ranking order
    but not probability magnitude.  Calibration (via CalibratedClassifierCV)
    would be needed for decision-theoretic use cases.
    """
    e = np.exp(Z - Z.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


# ---------------------------------------------------------------------------
# Regression evaluation
# ---------------------------------------------------------------------------
def evaluate_reg(model, X, y):
    """
    Compute regression metrics for any fitted scikit-learn regressor.

    Parameters
    ----------
    model : fitted regressor with .predict()
    X     : feature matrix
    y     : true target values

    Returns
    -------
    list[float]  [RMSE, MAE, R²]  — in the order defined by METRICS_REG.
    """
    p = model.predict(X)
    return [
        float(np.sqrt(mean_squared_error(y, p))),
        float(mean_absolute_error(y, p)),
        float(r2_score(y, p)),
    ]


# ---------------------------------------------------------------------------
# Classification evaluation
# ---------------------------------------------------------------------------
def evaluate_model(model, X, y):
    """
    Compute classification metrics for any fitted classifier.

    Handles both models that expose predict_proba (RandomForest, LightGBM)
    and those that only expose decision_function (LinearSVC).  This ensures
    ROC-AUC is always computable regardless of model architecture — critical
    for fair multi-model comparisons.

    Metric choices
    --------------
    Accuracy     : straightforward, but can be misleading with class imbalance.
    Macro F1     : treats all three duration classes equally; penalises models
                   that sacrifice one class to boost overall accuracy.  Primary
                   selection criterion throughout this project.
    ROC-AUC (OVR): area under the curve via One-vs-Rest; less sensitive to
                   class boundaries than F1, good secondary confirmation.

    Parameters
    ----------
    model : fitted classifier with .predict() and either .predict_proba()
            or .decision_function()
    X     : feature matrix (same column order as training data)
    y     : true integer class labels (0, 1, 2)

    Returns
    -------
    list[float]  [Accuracy, Macro F1, ROC-AUC (OVR)]  — order == METRICS.
    """
    p   = model.predict(X)
    acc = accuracy_score(y, p)
    f1  = f1_score(y, p, average='macro')
    try:
        scores = model.predict_proba(X)
    except AttributeError:
        # LinearSVC has no predict_proba — use softmax-normalised decision scores
        scores = _softmax(model.decision_function(X))
    auc = roc_auc_score(y, scores, multi_class='ovr', average='macro')
    return [float(acc), float(f1), float(auc)]


# ---------------------------------------------------------------------------
# Bootstrap confidence intervals
# ---------------------------------------------------------------------------
def bootstrap_ci(model, X, y_true, metric_fn=None, n_boot=500, ci=0.95,
                 seed=42):
    """
    Estimate a confidence interval for a scalar metric via percentile bootstrap.

    Uses the percentile method (Efron, 1979): resample rows with replacement,
    compute the metric on each resample, and take quantiles as CI bounds.
    Unlike normal-theory CIs, this approach makes no distributional assumptions
    — appropriate when class labels are data-driven (tertile binning) and the
    sampling distribution of the metric is unknown.

    Parameters
    ----------
    model     : fitted classifier with .predict()
    X         : feature matrix (pd.DataFrame or np.ndarray)
    y_true    : true labels
    metric_fn : callable(y_true, y_pred) -> float
                Default: macro F1.  Pass accuracy_score or any sklearn metric.
    n_boot    : number of bootstrap resamples (500 gives stable 95% CI estimates)
    ci        : confidence level (0.95 → 95% CI)
    seed      : random seed for reproducibility

    Returns
    -------
    (mean, lower_bound, upper_bound) as floats

    Example
    -------
    mean_f1, lo, hi = bootstrap_ci(best_rf, X_test, y_test)
    print(f'Macro F1: {mean_f1:.4f}  95% CI: [{lo:.4f}, {hi:.4f}]')
    """
    if metric_fn is None:
        metric_fn = lambda yt, yp: f1_score(yt, yp, average='macro')

    rng   = np.random.default_rng(seed)
    preds = model.predict(X)
    y_arr = np.asarray(y_true)
    n     = len(y_arr)

    boot = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        try:
            boot.append(metric_fn(y_arr[idx], preds[idx]))
        except Exception:
            pass   # skip degenerate resamples (single class present)

    boot = np.array(boot)
    lo   = np.percentile(boot, 100 * (1 - ci) / 2)
    hi   = np.percentile(boot, 100 * (1 + ci) / 2)
    return float(boot.mean()), float(lo), float(hi)
