import pandas as pd
import numpy as np


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
    
    return chunk[valid_mask].copy()




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
