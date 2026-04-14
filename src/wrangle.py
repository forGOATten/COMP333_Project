"""
wrangle.py — Data acquisition and cleaning for the Bixi-Weather pipeline.

Public API
----------
load_or_wrangle(processed_path, raw_bixi, raw_w1, raw_w2) -> pd.DataFrame
wrangle_bixi_chunk(chunk)                                  -> pd.DataFrame
load_and_filter_weather(path1, path2)                      -> pd.DataFrame
"""

import os
import pandas as pd
import numpy as np


# ---------------------------------------------------------------------------
# Pipeline entry point
# ---------------------------------------------------------------------------

def load_or_wrangle(processed_path, raw_bixi, raw_w1, raw_w2):
    """
    Return the cleaned Bixi-weather dataset from cache or full pipeline.

    Cache-or-compute pattern:
      - If processed_path exists  → load in seconds (avoids re-processing 3 GB).
      - If not                    → run chunked wrangling + weather join + save.

    To reproduce on a new Bixi season:
      1. Update the file-path constants in the notebook Setup cell.
      2. Delete the processed cache CSV.
      3. Re-run — this function rebuilds automatically.

    Parameters
    ----------
    processed_path : str   Path to the cached processed CSV.
    raw_bixi       : str   Path to the raw Bixi trip CSV (~3 GB).
    raw_w1         : str   Path to the first hourly weather CSV (Jan-Mar).
    raw_w2         : str   Path to the second hourly weather CSV (Mar-Dec).

    Returns
    -------
    pd.DataFrame  Cleaned, merged Bixi + Weather dataset.

    Notes
    -----
    - Bixi CSV is read in 100K-row chunks to avoid loading 11M+ rows into RAM.
    - Trips are joined to weather on the hour-rounded start time (left join).
    - Trips with no matching weather record are dropped (dropna on TEMP).
    """
    if os.path.exists(processed_path):
        print(f'Cache found -> {processed_path}')
        df = pd.read_csv(processed_path)
        print(f'Loaded: {df.shape[0]:,} rows x {df.shape[1]} columns')
        return df

    print('No cache found -- running full wrangling pipeline (several minutes)...')
    chunks = []
    for chunk in pd.read_csv(raw_bixi, chunksize=100_000):
        chunks.append(wrangle_bixi_chunk(chunk))
    df_bixi = pd.concat(chunks, ignore_index=True)

    df_w = load_and_filter_weather(raw_w1, raw_w2)
    df_w['LOCAL_DATE'] = pd.to_datetime(df_w['LOCAL_DATE']).dt.tz_localize(None)

    df = pd.merge(
        df_bixi, df_w,
        left_on='hour_rounded', right_on='LOCAL_DATE',
        how='left'
    )
    df.drop(columns=['LOCAL_DATE'], inplace=True)
    df = df.dropna(subset=['TEMP']).copy()

    print(f'Pipeline complete: {df.shape[0]:,} trips.')
    df.to_csv(processed_path, index=False)
    print(f'Saved -> {processed_path}')
    return df


# ---------------------------------------------------------------------------
# Chunk-level Bixi wrangling
# ---------------------------------------------------------------------------

#   Method cleans a chunk of Bixi data:
#    1. Handles 13-digit Unix ms timestamps.
#    2. Drops rows with missing critical data.
#    3. Filters outliers (Ghost trips < 5m, Forgotten trips > 240m).
def wrangle_bixi_chunk(chunk):
    critical_cols = [
        'STARTTIMEMS', 'ENDTIMEMS', 
        'STARTSTATIONLATITUDE', 'STARTSTATIONLONGITUDE',
        'ENDSTATIONLATITUDE', 'ENDSTATIONLONGITUDE'
    ]
    chunk = chunk.dropna(subset=critical_cols).copy()

    # 1. Convert Unix MS to UTC Datetime
    chunk['start_dt'] = pd.to_datetime(chunk['STARTTIMEMS'], unit='ms', utc=True)
    chunk['end_dt'] = pd.to_datetime(chunk['ENDTIMEMS'], unit='ms', utc=True)
    
    # 2. Convert UTC to Montreal Time and STRIP timezone info (Make it 'naive')
    # This is the "Magic Fix" that aligns it with the Weather CSV format
    chunk['start_dt'] = chunk['start_dt'].dt.tz_convert('America/Toronto').dt.tz_localize(None)
    chunk['end_dt'] = chunk['end_dt'].dt.tz_convert('America/Toronto').dt.tz_localize(None)
    
    # 3. Create the join key
    chunk['hour_rounded'] = chunk['start_dt'].dt.round('h')
    
    # 4. Duration and Outlier Filter
    chunk['duration_min'] = (chunk['end_dt'] - chunk['start_dt']).dt.total_seconds() / 60
    valid_mask = (chunk['duration_min'] >= 5) & (chunk['duration_min'] <= 240)
    
    chunk = chunk[valid_mask].copy()
    # Drop original ms-epoch columns — start_dt/end_dt already contain this info
    chunk = chunk.drop(columns=['STARTTIMEMS', 'ENDTIMEMS'], errors='ignore')
    return chunk




# Method that combines weather files and restricts data to exactly Jan 1 - Dec 31, 2025.
def load_and_filter_weather(path1, path2):
    w1 = pd.read_csv(path1)
    w2 = pd.read_csv(path2)
    
    # Combining and dropping any overlapping rows)
    df = pd.concat([w1, w2], ignore_index=True)
    
    # Ensuring LOCAL_DATE is parsed correctly without timezone 
    df['LOCAL_DATE'] = pd.to_datetime(df['LOCAL_DATE'])
    
    # Dropping duplicates in case the two files overlap on certain days
    df = df.drop_duplicates(subset=['LOCAL_DATE'])
    
    cols_to_keep = ['LOCAL_DATE', 'TEMP', 'PRECIP_AMOUNT', 'WINDCHILL']
    df = df[cols_to_keep]
    
    # Filtering for 2025
    mask = (df['LOCAL_DATE'] >= '2025-01-01') & (df['LOCAL_DATE'] <= '2025-12-31 23:59:59')
    
    return df[mask].sort_values('LOCAL_DATE').reset_index(drop=True)
