import numpy as np
import pandas as pd

from typing import Union, Optional,Tuple




def get_quick_summary( dataset : pd.DataFrame,\
                      *,
                      unique_ratio : float = 0.005,
                      distrib_range : Tuple[float,float] = (-0.3,0.3),
                      kurt_range : Tuple[float,float]= (3.0,3.0),
                      classify : bool = False,
                      ):

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
        _temp["class_len"] = _temp["features"].map(lambda c: unique_counts[c] if c in categorical_features else None)

        return _temp
    
        


    else:
        raise ValueError('Dataset cannot be empty! Please pass a valid dataset.')

