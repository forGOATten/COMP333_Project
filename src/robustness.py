"""
robustness.py — Comprehensive evaluation and robustness analysis functions.

Encapsulates all post-selection analysis so the notebook's Section 6 is
reduced to clean function calls, keeping the narrative focused on results.

Public API
----------
plot_learning_curves(model_class, model_params, X_tune, y_tune, ...)
temporal_robustness(model, X_test, y_test, ...)
subgroup_analysis(model, X_test, y_test, ...)
plot_bootstrap_ci(model_map, X_test, y_test, ...)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve, StratifiedKFold
from sklearn.metrics import f1_score, accuracy_score
from IPython.display import display

from evaluation import bootstrap_ci


# ---------------------------------------------------------------------------
# 6.1 Learning Curves
# ---------------------------------------------------------------------------
def plot_learning_curves(model_class, model_params, X_tune, y_tune,
                         seed=42, n_sizes=6, cv_folds=3, sample_size=100_000):
    """
    Plot training and cross-validation Macro F1 across increasing training sizes.

    Diagnoses whether the best model is in a high-bias (underfitting) or
    high-variance (overfitting) regime, and whether more data would help.

    Parameters
    ----------
    model_class  : sklearn estimator class (e.g. RandomForestClassifier)
    model_params : dict  best hyperparameters from the tuning search
    X_tune, y_tune : tuning sample (used for CV — kept tractable)
    seed         : random seed
    n_sizes      : number of training-size checkpoints
    cv_folds     : stratified k-fold splits
    sample_size  : rows drawn from X_tune (default 100K for speed)

    Returns
    -------
    None  (renders plot inline)
    """
    n = min(sample_size, len(X_tune))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(X_tune), size=n)
    X_lc = X_tune.iloc[idx].reset_index(drop=True)
    y_lc = y_tune.iloc[idx].reset_index(drop=True)

    print(f'Computing learning curves ({n:,} rows, {cv_folds}-fold CV)...')
    train_sizes, tr_scores, val_scores = learning_curve(
        model_class(**model_params, random_state=seed, n_jobs=-1),
        X_lc, y_lc,
        train_sizes=np.linspace(0.10, 1.0, n_sizes),
        cv=StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed),
        scoring='f1_macro',
        n_jobs=-1,
    )

    tr_mean,  tr_std  = tr_scores.mean(1),  tr_scores.std(1)
    val_mean, val_std = val_scores.mean(1), val_scores.std(1)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(train_sizes, tr_mean,  'o-', color='#2a6496', lw=2, label='Train F1')
    ax.fill_between(train_sizes, tr_mean - tr_std, tr_mean + tr_std,
                    alpha=0.15, color='#2a6496')
    ax.plot(train_sizes, val_mean, 's-', color='#e74c3c', lw=2, label='CV Val F1')
    ax.fill_between(train_sizes, val_mean - val_std, val_mean + val_std,
                    alpha=0.15, color='#e74c3c')
    ax.set_xlabel('Training examples')
    ax.set_ylabel('Macro F1')
    ax.set_title('Learning Curves — Best Model (Engineered Features)', fontweight='bold')
    ax.legend()
    ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.show()

    gap   = tr_mean[-1] - val_mean[-1]
    slope = val_mean[-1] - val_mean[0]
    print(f'Train–Val gap at full size : {gap:.4f}')
    print(f'Val F1 gain across curve   : {slope:+.4f}')
    if gap > 0.05:
        print('Interpretation: large gap → high-variance; consider stronger regularisation.')
    elif slope > 0.01:
        print('Interpretation: val F1 still rising → more data would likely help.')
    else:
        print('Interpretation: flat val curve → performance plateau; richer features needed.')


# ---------------------------------------------------------------------------
# 6.2 Temporal Robustness
# ---------------------------------------------------------------------------
def temporal_robustness(model, X_test, y_test):
    """
    Compute and plot Macro F1 broken down by calendar month on the test set.

    Reveals whether model performance degrades in specific months (e.g.
    shoulder season, winter) — a failure mode that aggregate metrics hide.

    Requires X_test to contain a 'month' column (present in FEATURES_ENG_FINAL).

    Parameters
    ----------
    model         : fitted classifier
    X_test, y_test: held-out test split

    Returns
    -------
    pd.Series  monthly Macro F1 scores (index = month abbreviation)
    """
    month_names = {
        1:'Jan', 2:'Feb', 3:'Mar', 4:'Apr', 5:'May', 6:'Jun',
        7:'Jul', 8:'Aug', 9:'Sep', 10:'Oct', 11:'Nov', 12:'Dec',
    }
    preds   = model.predict(X_test)
    y_arr   = np.asarray(y_test)
    months  = X_test['month'].values

    monthly = {}
    for m in sorted(np.unique(months)):
        mask = months == m
        if mask.sum() >= 100:
            monthly[month_names.get(int(m), str(m))] = f1_score(
                y_arr[mask], preds[mask], average='macro'
            )
    s = pd.Series(monthly)

    colors = [
        '#c0392b' if v < s.mean() - s.std()
        else '#27ae60' if v > s.mean() + s.std()
        else '#2a6496'
        for v in s.values
    ]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(s.index, s.values, color=colors, edgecolor='white')
    ax.axhline(s.mean(), color='black', ls='--', lw=1.2,
               label=f'Mean F1 = {s.mean():.3f}')
    ax.axhline(0.333, color='gray', ls=':', lw=1, label='Random baseline')
    ax.set_ylabel('Macro F1')
    ax.set_title('Month-by-Month Macro F1 — Best Model (Test Set)', fontweight='bold')
    ax.set_ylim(0, 0.75)
    ax.legend()
    for i, (m, v) in enumerate(s.items()):
        ax.text(i, v + 0.01, f'{v:.3f}', ha='center', fontsize=8)
    plt.tight_layout()
    plt.show()

    print(f'Monthly F1 range : {s.min():.3f} – {s.max():.3f}')
    print(f'Std across months: {s.std():.4f}')
    if s.std() > 0.05:
        print('High variance → model is temporally inconsistent.')
    else:
        print('Low variance → model generalises robustly across months.')
    return s


# ---------------------------------------------------------------------------
# 6.3 Subgroup Analysis
# ---------------------------------------------------------------------------
def subgroup_analysis(model, X_test, y_test):
    """
    Compute Macro F1 and Accuracy across condition-based subgroups.

    Checks whether aggregate performance masks disparities across identifiable
    rider contexts: weather conditions, time of day, and day type.

    Standard subgroups (derived from FEATURES_ENG_FINAL columns):
        Warm / Cool, Rainy / Dry, Rush Hour / Off-Peak, Weekend / Weekday

    Parameters
    ----------
    model         : fitted classifier
    X_test, y_test: held-out test split

    Returns
    -------
    pd.DataFrame  subgroup performance table with Delta vs Overall column
    """
    preds  = model.predict(X_test)
    y_arr  = np.asarray(y_test)
    overall_f1 = f1_score(y_arr, preds, average='macro')

    subgroups = {
        'Warm (TEMP > 15°C)' : (X_test['TEMP'] > 15).values,
        'Cool (TEMP ≤ 15°C)' : (X_test['TEMP'] <= 15).values,
        'Rainy (PRECIP > 0)' : (X_test['PRECIP_AMOUNT'] > 0).values,
        'Dry   (PRECIP = 0)' : (X_test['PRECIP_AMOUNT'] == 0).values,
        'Rush Hour'          : (X_test['is_rush_hour'] == 1).values,
        'Off-Peak'           : (X_test['is_rush_hour'] == 0).values,
        'Weekend'            : (X_test['is_weekend'] == 1).values,
        'Weekday'            : (X_test['is_weekend'] == 0).values,
    }

    rows = []
    for name, mask in subgroups.items():
        if mask.sum() >= 500:
            rows.append({
                'Subgroup'        : name,
                'N'               : int(mask.sum()),
                'Macro F1'        : round(f1_score(y_arr[mask], preds[mask], average='macro'), 4),
                'Accuracy'        : round(accuracy_score(y_arr[mask], preds[mask]), 4),
                'Δ vs Overall'    : round(
                    f1_score(y_arr[mask], preds[mask], average='macro') - overall_f1, 4
                ),
            })

    df = pd.DataFrame(rows).set_index('Subgroup')
    display(df.style
        .map(
            lambda v: ('background-color:#f8d7da' if isinstance(v, float) and v < -0.02
                       else 'background-color:#d4edda' if isinstance(v, float) and v > 0.02
                       else ''),
            subset=['Δ vs Overall']
        )
        .format({'Macro F1': '{:.4f}', 'Accuracy': '{:.4f}',
                 'Δ vs Overall': '{:+.4f}', 'N': '{:,}'})
        .set_caption(
            'Subgroup performance on test set. '
            'Green = outperforms overall by >0.02 F1; Red = underperforms.'
        ))
    return df


# ---------------------------------------------------------------------------
# 6.4 Bootstrap Confidence Intervals
# ---------------------------------------------------------------------------
def plot_bootstrap_ci(model_map, X_test, y_test,
                      n_boot=500, sample_size=200_000, seed=42):
    """
    Compute and visualise bootstrap 95% CIs for all engineered models.

    Uses percentile bootstrap (Efron, 1979) on a subsample of the test set.
    Renders both a styled summary table and a forest plot.

    Parameters
    ----------
    model_map   : dict  {model_name: fitted_model}
    X_test, y_test: held-out test split
    n_boot      : bootstrap resamples (500 → stable 95% CI)
    sample_size : rows sampled from X_test (200K for speed)
    seed        : random seed

    Returns
    -------
    pd.DataFrame  CI summary table
    """
    n   = min(sample_size, len(X_test))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(X_test), size=n)
    X_ci = X_test.iloc[idx]
    y_ci = y_test.iloc[idx]

    print(f'Computing bootstrap 95% CIs ({n:,} subsample, {n_boot} resamples)...')
    rows = []
    for name, model in model_map.items():
        mean_f1, lo_f1, hi_f1 = bootstrap_ci(
            model, X_ci, y_ci,
            metric_fn=lambda yt, yp: f1_score(yt, yp, average='macro'),
            n_boot=n_boot, seed=seed,
        )
        mean_acc, lo_acc, hi_acc = bootstrap_ci(
            model, X_ci, y_ci,
            metric_fn=accuracy_score,
            n_boot=n_boot, seed=seed,
        )
        rows.append({
            'Model'      : name,
            'F1 Mean'    : round(mean_f1,  4),
            'F1 95% CI'  : f'[{lo_f1:.4f}, {hi_f1:.4f}]',
            'F1 Width'   : round(hi_f1 - lo_f1, 4),
            'Acc Mean'   : round(mean_acc, 4),
            'Acc 95% CI' : f'[{lo_acc:.4f}, {hi_acc:.4f}]',
        })

    df = pd.DataFrame(rows).set_index('Model')
    display(df.style
        .background_gradient(subset=['F1 Mean', 'Acc Mean'], cmap='Greens')
        .background_gradient(subset=['F1 Width'], cmap='Reds_r')
        .set_caption(
            f'Bootstrap 95% CIs ({n_boot} resamples, {n:,} test subsample). '
            'Narrower F1 Width = more stable estimate.'
        ))

    # Forest plot
    fig, ax = plt.subplots(figsize=(9, max(3, len(rows) * 1.2)))
    for i, row in enumerate(rows):
        lo = float(row['F1 95% CI'].split(',')[0].strip('['))
        hi = float(row['F1 95% CI'].split(',')[1].strip(']'))
        ax.plot([lo, hi], [i, i], 'o-', color='#2a6496', lw=2.5, ms=6)
        ax.text(hi + 0.001, i, f"{row['F1 Mean']:.4f}", va='center', fontsize=8)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r['Model'] for r in rows], fontsize=9)
    ax.set_xlabel('Macro F1')
    ax.set_title('Bootstrap 95% CIs — Engineered Models (Test Set)', fontweight='bold')
    ax.axvline(0.333, color='gray', ls=':', lw=1, label='Random baseline')
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.show()

    return df
