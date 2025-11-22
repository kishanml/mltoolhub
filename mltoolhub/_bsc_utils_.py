import math
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from typing import Union, Optional,Tuple

def reduce_memory_usage(dataset, downsample=False, fraction=0.5):
    
    """
    Reduce memory usage of a dataframe by downcasting numeric columns.
    
    Parameters:
        dataset (pd.DataFrame): Input dataframe
        downsample (bool): If True, sample a fraction of rows
        fraction (float): Fraction of rows to keep if downsampling (0 < fraction <= 1)
        
    Returns:
        pd.DataFrame: Reduced memory dataframe

    """
    
    df = dataset.copy()
    
    if downsample and 0 < fraction < 1:
        df = df.sample(frac=fraction, random_state=42).reset_index(drop=True)
    
    for col in df.select_dtypes(include=['int', 'int64']).columns:

        c_min = df[col].min()
        c_max = df[col].max()
        if c_min >= np.iinfo(np.int8).min and c_max <= np.iinfo(np.int8).max:
            df[col] = df[col].astype(np.int8)
        elif c_min >= np.iinfo(np.int16).min and c_max <= np.iinfo(np.int16).max:
            df[col] = df[col].astype(np.int16)
        elif c_min >= np.iinfo(np.int32).min and c_max <= np.iinfo(np.int32).max:
            df[col] = df[col].astype(np.int32)
        else:
            df[col] = df[col].astype(np.int64) 
    
    for col in df.select_dtypes(include=['float', 'float64']).columns:
        df[col] = df[col].astype(np.float16)
    
    return df

def get_quick_summary( dataset : pd.DataFrame,\
                      *,
                      unique_ratio : float = 0.005,
                      distrib_range : Tuple[float,float] = (-0.3,0.3),
                      kurt_range : Tuple[float,float]= (3.0,3.0),
                      classify : bool = False,
                      ) -> pd.DataFrame:

    if dataset.size:

        observations_len, _  = dataset.shape

        _temp = dataset.dtypes.reset_index().rename(columns={"index":'features',0:'dtypes'})

        # feature's missing values
        missing_values = dataset.isna().sum()
        _temp['missing_count'] = _temp['features'].map(missing_values)
        _temp['missing_percentage'] = ( _temp['missing_count']/ observations_len ) * 100

        # numeric or categorical 
        object_types = _temp.loc[_temp['dtypes']=='object','features'].to_list()

        numeric_types = _temp.loc[(_temp['dtypes']!='object') & (_temp['missing_percentage']<75),'features']
        uniqueness_ratio = dataset[numeric_types].nunique()/observations_len
        expected_object_types = expected_object_types = uniqueness_ratio[uniqueness_ratio < unique_ratio].index.to_list()

        categorical_features = object_types + expected_object_types
        true_numerical_features = dataset.columns.difference(categorical_features).to_list()
        _temp['nature'] = np.where(_temp['features'].isin(categorical_features),'category','numeric')

        # if numeric, distribution
        _skew = dataset[true_numerical_features].skew()
        _temp['skewness'] = _temp['features'].map(lambda c : _skew[c] if c in _skew else None)
        if classify:
            classify_skew = lambda val : "right-skewed" if val> distrib_range[1] else ("left-skewed" if val< distrib_range[0] else "normal")
            _temp['skew_type'] = _temp['skewness'].apply(classify_skew)

        # if numeric, outlier presence
        _kurt = dataset[true_numerical_features].kurt()
        _temp['kurtosis'] = _temp['features'].map(lambda c : _kurt[c] if c in _skew else None)

        if classify:
            classify_kurt = lambda val : "lepto" if val > kurt_range[1] else ("platy" if val< kurt_range[0] else "meso")
            _temp['kurt_type'] = _temp['kurtosis'].apply(classify_kurt)

        # if categorical, counts
        unique_counts = dataset.nunique()
        _temp["no_of_classes"] = _temp["features"].map(lambda c: unique_counts[c] if c in categorical_features else None)

        return _temp
    

    else:
        raise ValueError('Dataset cannot be empty! Please pass a valid dataset.')

def get_summary_plots(dataset : pd.DataFrame, *, max_height : int = 12):
   
    
    sns.set(style="whitegrid")
    figs = []

    _summary = get_quick_summary(dataset,classify=True)

    # 1. Missing percentage per feature
    temp_missing = _summary.loc[_summary['missing_percentage'] != 0,['features', 'missing_count', 'missing_percentage']].sort_values(by='missing_percentage')

    if len(temp_missing) > 0:
        fig_height = min(max_height, 0.45 * len(temp_missing))
        fig1, ax1 = plt.subplots(figsize=(12, fig_height))
        
        sns.barplot(
            data=temp_missing,
            x='missing_percentage',
            y='features',
            color="#4DA6FF",
            ax=ax1
        )

        for i, (pct, count) in enumerate(zip(temp_missing['missing_percentage'], temp_missing['missing_count'])):
            ax1.text(
                pct + 0.5,
                i,
                str(count),
                va='center',
                ha='left',
                fontsize=10,
                fontweight='bold'
            )

        ax1.set_xlabel("Missing Percentage (%)", fontsize=14)
        ax1.set_ylabel("")
        ax1.invert_yaxis()

        fig1.suptitle("Missing Percentage per Feature", fontsize=16, y=1.02)
        fig1.tight_layout()
        figs.append(fig1)


    # 2. Histograms of numeric features (with skewness)
    temp_numeric = _summary.loc[_summary['nature'] == 'numeric', ['features', 'skewness', 'skew_type','kurt_type']]
    n = len(temp_numeric)
    if n > 0:
        cols = 5
        rows = math.ceil(n / cols)
        fig2, axes2 = plt.subplots(rows, cols, figsize=(max_height, 3*rows) ,dpi=80)
        axes2 = axes2.flatten()

        for i, ax in enumerate(axes2):
            if i < n:
                feature = temp_numeric.iloc[i, 0]
                skew_type = temp_numeric.iloc[i, 2]

                sns.histplot(
                    data=dataset,
                    x=feature,
                    bins=30,
                    kde=True,
                    color='steelblue',
                    ax=ax
                )
                ax.set_title(f"{feature} ({skew_type})", fontsize=10)
            else:
                ax.axis("off")
        fig2.suptitle("Histograms of Numeric Features (Skewness)", fontsize=16, y=1.02)
        fig2.tight_layout()
        figs.append(fig2)


    # 3. Boxen plots of numeric features (with kurtosis)
    n = len(temp_numeric)
    if n > 0:
        cols = 5
        rows = math.ceil(n / cols)
        fig3, axes3 = plt.subplots(rows, cols, figsize=(max_height, 3*rows),dpi=80)
        axes3 = axes3.flatten()

        for i, ax in enumerate(axes3):
            if i < n:
                feature = temp_numeric.iloc[i, 0]
                kurt_type = temp_numeric.iloc[i, -1]

                sns.boxenplot(x=dataset[feature], ax=ax)
                ax.set_title(f"{feature} ({kurt_type})", fontsize=10)
            else:
                ax.axis("off")
        fig3.suptitle("Boxen Plots of Numeric Features (Kurtosis)", fontsize=16, y=1.02)
        fig3.tight_layout()
        figs.append(fig3)


    # 4. Value counts for categorical features
    temp_cat = _summary.loc[_summary['nature'] == 'category', 'features']
    n = len(temp_cat)
    if n > 0:
        cols = 3
        rows = math.ceil(n / cols)
        fig4, axes4 = plt.subplots(rows, cols, figsize=(max_height, 3*rows),dpi=80)
        axes4 = axes4.flatten()

        for i, ax in enumerate(axes4):
            if i < n:
                feature = temp_cat.iloc[i]

                sns.countplot(
                    data=dataset,
                    x=feature,
                    ax=ax,
                    palette="Blues_r"
                )

                ax.set_title(feature, fontsize=10)
                ax.set_xlabel("")
                ax.set_ylabel("Count")
                ax.tick_params(axis='x', rotation=45)
            else:
                ax.axis("off")
        fig4.suptitle("Value Counts for Categorical Features", fontsize=16, y=1.02)
        fig4.tight_layout()
        figs.append(fig4)

    return figs
