import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def quantDDA(df):    
    rows = []
    for col in df.columns:
        column = df[col]
        columnNum = column.dropna()  

        rowsTotal = len(column)
        rowsEntries = column.notna().sum()
        rowsMissing = column.isna().sum()
        rowsUnique = column.nunique()
        
        modes = column.mode()
        modesString = ", ".join(map(str, modes.tolist())) if not modes.empty else "None"

        mean = std = maxValue = minValue = q1 = q2 = q3 = skew = kurt = outliers = np.nan
        
        if pd.api.types.is_numeric_dtype(column):
            mean = columnNum.mean()
            std = columnNum.std()
            maxValue = columnNum.max()
            minValue = columnNum.min()
            q1, q2, q3 = columnNum.quantile([0.25, 0.5, 0.75])
            skew = columnNum.skew()
            kurt = columnNum.kurt()
            iqr = q3 - q1
            outliers = ((columnNum < (q1 - 1.5 * iqr)) | (columnNum > (q3 + 1.5 * iqr))).sum()
            
        rows.append({
            "Feature": col,
            "Total Obs": rowsTotal,
            "Missing": rowsMissing,
            "Unique": rowsUnique,
            "Outliers (IQR)": outliers,
            "Mode": modesString,
            "Mean": mean,
            "StdDev": std,
            "Min": minValue,
            "Median": q2,
            "Max": maxValue,
            "Skew": skew
        })
    return pd.DataFrame(rows).set_index("Feature")


def vizDDA(df, sample_size=50000):
    if len(df) > sample_size:
        plot_df = df.sample(n=sample_size, random_state=42).copy()
    else:
        plot_df = df.copy()

    # ---  Seasonal Grouping ---
    if 'start_dt' in plot_df.columns:
        plot_df['start_dt'] = pd.to_datetime(plot_df['start_dt'])
        plot_df['Month'] = plot_df['start_dt'].dt.month
        def get_season(m):
            if 1 <= m <= 3: return "Q1: Winter (Jan-Mar)"
            if 4 <= m <= 6: return "Q2: Spring (Apr-Jun)"
            if 7 <= m <= 9: return "Q3: Summer (Jul-Sep)"
            return "Q4: Fall (Oct-Dec)"
        plot_df['Season'] = plot_df['Month'].apply(get_season)

    # ---  Weather vs. Duration Grid ---
    fig, axes = plt.subplots(1, 3, figsize=(20, 5))

    #  Temperature (range: -30 to 30)
    sns.scatterplot(data=plot_df, x='TEMP', y='duration_min', alpha=0.1, ax=axes[0], color='orange')
    axes[0].set_title("Temp vs Duration")
    axes[0].set_xlim(-30, 30)
    axes[0].set_ylabel("Duration (min)")

    # Windchill (range: -35 to 0)
    # Note: We reverse the limits to show -35 on the left and 0 on the right
    sns.scatterplot(data=plot_df, x='WINDCHILL', y='duration_min', alpha=0.1, ax=axes[1], color='blue')
    axes[1].set_title("Windchill vs Duration")
    axes[1].set_xlim(-35, 0)
    axes[1].set_ylabel("Duration (min)")

    # Precipitation (Requested range: 0 to 30)
    sns.scatterplot(data=plot_df, x='PRECIP_AMOUNT', y='duration_min', alpha=0.1, ax=axes[2], color='navy')
    axes[2].set_title("Precipitation vs Duration")
    axes[2].set_xlim(0, 30)
    axes[2].set_ylabel("Duration (min)")

    plt.tight_layout()
    plt.show()

    # Seasonal Boxplot
    if 'Season' in plot_df.columns:
        plt.figure(figsize=(12, 6))
        season_order = ["Q1: Winter (Jan-Mar)", "Q2: Spring (Apr-Jun)", 
                        "Q3: Summer (Jul-Sep)", "Q4: Fall (Oct-Dec)"]
        sns.boxplot(data=plot_df, x='Season', y='duration_min', hue='Season', 
                    order=season_order, palette='Set2', legend=False)
        plt.title("Seasonal Trip Duration Trends")
        plt.ylabel("Duration (min)")
        plt.show()

    # Correlation  
    plt.figure(figsize=(10, 4))
    corr_cols = [c for c in ['duration_min', 'TEMP', 'PRECIP_AMOUNT', 'WINDCHILL'] if c in plot_df.columns]
    sns.heatmap(plot_df[corr_cols].corr(), annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Correlation Matrix")
    plt.show()

    # Missing Values
    plt.figure(figsize=(10, 2))
    # We remove yticklabels to keep the heatmap clean
    sns.heatmap(plot_df.isnull(), cbar=False, cmap='viridis', yticklabels=False)
    plt.title("Final Missing Values Audit (Uniform color = Success)")
    plt.show()
