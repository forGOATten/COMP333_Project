"""
features.py — Feature engineering pipeline for the Bixi-Weather dataset.

Encapsulates all transformations applied between the raw merged dataset and
the model input matrix so that:
  1. The notebook stays focused on analysis, not implementation.
  2. The same pipeline can be applied identically to training, validation,
     test, and new season data — preventing data leakage.

Public API
----------
engineer_features(df, bins_qcut=None) -> (df_eng, bins, feature_names)
FEATURES_ENG_FINAL                   -> list[str]  (15 feature names)
"""

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Feature name registry
# Defines the canonical order expected by all downstream models.
# Any future feature addition must be reflected here first.
# ---------------------------------------------------------------------------
FEATURES_ENG_FINAL = [
    # Weather (raw)
    'TEMP', 'PRECIP_AMOUNT', 'WINDCHILL',
    # Temporal (extracted from datetime index)
    'hour', 'day_of_week', 'month', 'is_weekend',
    # Cyclical encodings (maps hour/month onto unit circle)
    'hour_sin', 'hour_cos', 'month_sin', 'month_cos',
    # Interactions & polynomial
    'temp_weekend', 'precip_rush_hour', 'temp_squared', 'is_rush_hour',
]


def engineer_features(df, bins_qcut=None):
    """
    Apply the full 15-feature engineering pipeline to a Bixi-weather dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain: hour_rounded, TEMP, PRECIP_AMOUNT, duration_min.
        WINDCHILL is optional; falls back to TEMP when absent (correct at T > 0°C
        per Environment Canada methodology).

    bins_qcut : array-like or None
        Tertile bin edges computed on the *training set*.
        - Pass None on training data  → edges computed via pd.qcut (data-driven).
        - Pass BINS_QCUT on test/new data → edges fixed to training distribution.
        This prevents data leakage: class boundaries must never be influenced by
        unseen data.

    Returns
    -------
    df_eng   : pd.DataFrame    Enriched dataframe containing all engineered columns.
    bins     : np.ndarray      Tertile bin edges. Store as BINS_QCUT and pass to
                               any subsequent call on held-out or new data.
    features : list[str]       Ordered feature names (== FEATURES_ENG_FINAL).

    Raises
    ------
    ValueError  If any required column is missing. Fast-fail prevents silent
                errors from propagating into model training.

    Notes
    -----
    Feature groups and rationale:

    Temporal  — hour, day_of_week, month, is_weekend
        Rider intent (commute vs. leisure) is largely determined by when the trip
        starts, not by the weather.  These four features encode that context.

    Cyclical  — hour_sin/cos, month_sin/cos
        Linear encoding of hour (0-23) treats 23 and 0 as far apart; sine/cosine
        projection onto the unit circle makes them adjacent.  Same logic for months.

    Interaction — temp_weekend, precip_rush_hour
        The effect of temperature on duration differs between weekdays and weekends;
        the effect of rain differs by time of day.  Explicit interaction terms let
        linear models capture these conditional relationships.

    Polynomial — temp_squared
        Temperature has a non-linear 'sweet spot' effect on cycling duration.
        Squaring captures the concave relationship without adding a full polynomial
        basis expansion.

    Domain — is_rush_hour
        Binary flag for peak commute hours (07-09, 16-18).  Separates purposeful
        short commutes from exploratory longer rides.

    WINDCHILL — fills NaN with TEMP
        Environment Canada computes windchill only when T <= 0°C and wind speed is
        sufficient.  Above freezing, perceived temperature equals air temperature,
        so TEMP is the correct fallback.
    """
    required = {'hour_rounded', 'TEMP', 'PRECIP_AMOUNT', 'duration_min'}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(
            f'engineer_features(): missing required columns: {missing}\n'
            f'Available columns: {sorted(df.columns.tolist())}'
        )

    df_eng = df.copy()
    df_eng['hour_rounded'] = pd.to_datetime(df_eng['hour_rounded'])

    # ── Temporal features ────────────────────────────────────────────────────
    df_eng['hour']        = df_eng['hour_rounded'].dt.hour
    df_eng['day_of_week'] = df_eng['hour_rounded'].dt.dayofweek   # 0=Mon, 6=Sun
    df_eng['month']       = df_eng['hour_rounded'].dt.month
    df_eng['is_weekend']  = (df_eng['day_of_week'] >= 5).astype(int)

    # ── Cyclical encodings ───────────────────────────────────────────────────
    df_eng['hour_sin']  = np.sin(2 * np.pi * df_eng['hour']  / 24)
    df_eng['hour_cos']  = np.cos(2 * np.pi * df_eng['hour']  / 24)
    df_eng['month_sin'] = np.sin(2 * np.pi * df_eng['month'] / 12)
    df_eng['month_cos'] = np.cos(2 * np.pi * df_eng['month'] / 12)

    # ── Interaction & polynomial features ───────────────────────────────────
    df_eng['temp_weekend']     = df_eng['TEMP'] * df_eng['is_weekend']
    df_eng['precip_rush_hour'] = df_eng['PRECIP_AMOUNT'] * df_eng['hour']
    df_eng['temp_squared']     = df_eng['TEMP'] ** 2
    df_eng['is_rush_hour']     = df_eng['hour'].isin(
        [7, 8, 9, 16, 17, 18]).astype(int)

    # ── WINDCHILL fallback ───────────────────────────────────────────────────
    if 'WINDCHILL' in df_eng.columns:
        df_eng['WINDCHILL'] = df_eng['WINDCHILL'].fillna(df_eng['TEMP'])
    else:
        df_eng['WINDCHILL'] = df_eng['TEMP']

    # ── Classification target (tertile binning) ──────────────────────────────
    if bins_qcut is None:
        df_eng['duration_cat'], bins_qcut = pd.qcut(
            df_eng['duration_min'], q=3, labels=[0, 1, 2], retbins=True
        )
    else:
        df_eng['duration_cat'] = pd.cut(
            df_eng['duration_min'], bins=bins_qcut, labels=[0, 1, 2],
            include_lowest=True
        )
    df_eng['duration_cat'] = df_eng['duration_cat'].astype(int)

    return df_eng, np.array(bins_qcut), FEATURES_ENG_FINAL
