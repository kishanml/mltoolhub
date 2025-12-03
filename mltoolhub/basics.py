import math
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')

from typing import Tuple

# -----------------------------------BASICS-------------------------------------

def get_quick_summary( dataset : pd.DataFrame,\
                      *,
                      unique_ratio : float = 0.002,
                      distrib_range : Tuple[float,float] = (-0.3,0.3),
                      kurt_range : Tuple[float,float]= (2.5,3.5),
                      classify : bool = False,
                      ) -> pd.DataFrame:
    """
    Usage
    -----
    Generate a quick summary of a pandas DataFrame with insights on missing values, 
    distribution, and outliers.
    
    Parameters
    ----------
        dataset : pd.DataFrame
            The input DataFrame to summarize.
        unique_ratio : float, optional, default=0.005
            Threshold ratio of unique values to total rows for flagging a column as 'mostly constant'.
        distrib_range : tuple of float, optional, default=(-0.3, 0.3)
            Expected range for the skewness of numerical columns. Columns outside this range may be flagged as skewed.
        kurt_range : tuple of float, optional, default=(3.0, 3.0)
            Expected range for the kurtosis of numerical columns. Columns outside this range may be flagged as having unusual tail behavior.
        classify : bool, optional, default=False
            If True, attempts to classify columns as 'categorical' or 'numerical' based on their dtype and unique values.

    Returns
    -------
        pd.DataFrame : dataset summary
    
    """
    _unknown = np.nan
    _df = dataset.copy()

    if _df.size:

        observations_len, _  = _df.shape

        _temp = _df.dtypes.reset_index().rename(columns={"index":'feature',0:'dtype'})

        # feature's missing values
        missing_values = _df.isna().sum()
        _temp['missing_count'] = _temp['feature'].map(missing_values)
        _temp['missing_percentage'] = ( _temp['missing_count']/ observations_len ) * 100

        # numeric / datetime /categorical features 
        _bool_feats = _df.select_dtypes(include="bool").columns
        _df[_bool_feats] = _df[_bool_feats].astype(np.int8)

        _num_feats = _df.select_dtypes(include=np.number).columns.tolist()
        _dt_feats = _df.select_dtypes(include=['datetime', 'datetimetz', 'timedelta']).columns.tolist()
        _obj_feats = _df.select_dtypes(include=['object', 'category']).columns.tolist()

        valid_numeric_feats = [col for col in _num_feats if col not in _dt_feats]

        # check if there any numeric features that are actually categorical.
        numeric_feats_to_validate = _temp.loc[(_temp['feature'].isin(valid_numeric_feats)) & (_temp['missing_percentage'] < 75), 'feature'].to_list()
        uniqueness_ratio = _df[numeric_feats_to_validate].nunique()/observations_len
        expected_object_feats  = uniqueness_ratio[uniqueness_ratio < unique_ratio].index.to_list()

        categorical_features = _obj_feats + expected_object_feats
        true_numerical_features = [feat for feat in valid_numeric_feats if feat not in expected_object_feats]

        _temp['nature'] = np.where(_temp['feature'].isin(categorical_features),'category',np.where(_temp['feature'].isin(true_numerical_features),'numeric','datetime'))

        # if numeric, distribution
        _skew = _df[true_numerical_features].skew()
        _temp['skewness'] = _temp['feature'].map(lambda c : _skew[c] if c in _skew else _unknown)
        if classify:
            classify_skew = lambda val : "right-skewed" if val> distrib_range[1] else ("left-skewed" if val< distrib_range[0] else ("normal" if distrib_range[0]<val<distrib_range[1] else _unknown))
            _temp['skew_type'] = _temp['skewness'].apply(classify_skew)

        # if numeric, outlier presence
        _kurt = _df[true_numerical_features].kurt()
        _temp['kurtosis'] = _temp['feature'].map(lambda c : _kurt[c] if c in _skew else _unknown)

        if classify:
            classify_kurt = lambda val : "lepto" if val > kurt_range[1] else ("platy" if val< kurt_range[0]  else ("meso" if kurt_range[0]<val<kurt_range[1] else _unknown))
            _temp['kurt_type'] = _temp['kurtosis'].apply(classify_kurt)

        # if categorical, counts
        unique_counts = _df.nunique()
        _temp["no_of_classes"] = _temp["feature"].map(lambda c: unique_counts[c] if c in categorical_features else _unknown)

        return _temp
    

    else:
        raise ValueError('Dataset cannot be empty! Please pass a valid dataset.')

def get_summary_plots(dataset : pd.DataFrame, \
                      *, 
                      max_dim : int = 12, \
                      sample_frac : float = 0.5,\
                      **kwargs) -> None:
   
    """
    Usage
    -----
    Generate summary plots for a pandas DataFrame to visualize distributions and data characteristics.

    Parameters
    ----------
        dataset : pd.DataFrame
            The input DataFrame to visualize.
        max_dim : int, optional, default=12
            Maximum height for the plots (useful for controlling figure size).

    Returns
    -------
        None
    """

    sns.set(style="whitegrid")

    dataset_sample = dataset.sample(frac=sample_frac).reset_index(drop=True)
    _summary = get_quick_summary(dataset_sample,classify=True,**kwargs)

    temp_missing = _summary.loc[_summary['missing_percentage'] != 0,['feature', 'missing_count', 'missing_percentage']].sort_values(by='missing_percentage')

    if len(temp_missing) > 0:
        fig_height = min(max_dim, 0.45 * len(temp_missing))
        fig1, ax1 = plt.subplots(figsize=(max_dim, fig_height))
        
        sns.barplot(
            data=temp_missing,
            x='missing_percentage',
            y='feature',
            color="#4DA6FF",
            ax=ax1
        )

        ax1.set_xlabel("Missing Percentage (%)", fontsize=10)
        ax1.set_ylabel("")
        ax1.invert_yaxis()

        fig1.suptitle("Missing Percentage per Feature", fontsize=16, y=1.02)
        fig1.tight_layout()


    # 2. Histograms of numeric features (with skewness)
    temp_numeric = _summary.loc[_summary['nature'] == 'numeric', ['feature', 'skewness', 'skew_type','kurt_type']]
    n = len(temp_numeric)
    if n > 0:
        cols = 5
        rows = math.ceil(n / cols)
        fig2, axes2 = plt.subplots(rows, cols, figsize=(max_dim, 3*rows) ,dpi=80)
        axes2 = axes2.flatten()

        for i, ax in enumerate(axes2):
            if i < n:
                feature = temp_numeric.iloc[i, 0]
                skew_type = temp_numeric.iloc[i, 2]

                sns.histplot(
                    data=dataset_sample,
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


    # 3. Boxen plots of numeric features (with kurtosis)
    n = len(temp_numeric)
    if n > 0:
        cols = 5
        rows = math.ceil(n / cols)
        fig3, axes3 = plt.subplots(rows, cols, figsize=(max_dim, 3*rows),dpi=80)
        axes3 = axes3.flatten()

        for i, ax in enumerate(axes3):
            if i < n:
                feature = temp_numeric.iloc[i, 0]
                kurt_type = temp_numeric.iloc[i, -1]

                sns.boxenplot(x=dataset_sample[feature], ax=ax)
                ax.set_title(f"{feature} ({kurt_type})", fontsize=10)
            else:
                ax.axis("off")
        fig3.suptitle("Boxen Plots of Numeric Features (Kurtosis)", fontsize=16, y=1.02)
        fig3.tight_layout()


    # 4. Value counts for categorical features
    temp_cat = _summary.loc[_summary['nature'] == 'category', 'feature']
    n = len(temp_cat)
    if n > 0:
        cols = 3
        rows = math.ceil(n / cols)
        fig4, axes4 = plt.subplots(rows, cols, figsize=(max_dim, 3*rows),dpi=80)
        axes4 = axes4.flatten()

        for i, ax in enumerate(axes4):
            if i < n:
                feature = temp_cat.iloc[i]

                sns.countplot(
                    data=dataset_sample,
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

