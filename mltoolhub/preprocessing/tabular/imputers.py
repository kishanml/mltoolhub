
# 
import numpy as np
import pandas as pd
from tqdm import tqdm

import warnings
warnings.filterwarnings('ignore')


import lightgbm as lgb
from basics import get_quick_summary
from sklearn.preprocessing import  OrdinalEncoder



class TabImputer:


    def __init__(self, no_of_iterations : int = 5,regressor : object = None, classifier : object = None, device : str = "cpu"):
        
        self._iterations = no_of_iterations


        LGB_PARAMS =dict(
            device='gpu' if device=='cuda' else device,             
            verbosity=-1
        )

        
        if regressor is None:
            self._regressor = lgb.LGBMRegressor(**LGB_PARAMS)
        else:
            self._regressor = regressor

        if classifier is None:
            self._classifier = lgb.LGBMClassifier(**LGB_PARAMS)
        else:
            self._classifier = classifier

    
        self._encoder = None



    def stats_impute(self,dataset : pd.DataFrame, summary : pd.DataFrame = None) -> pd.DataFrame:

        df = dataset.copy()

        if summary is None:
            summary = get_quick_summary(df,classify=True)

        for _,row in summary.iterrows():

            if row['nature'] == "numeric" and ( row['skew_type'] == "right-skewed" or row['skew_type'] == "left-skewed" ) :
                df[row['feature']] = df[row['feature']].fillna(df[row['feature']].median())
            elif row['nature'] == "numeric" and row['skew_type'] == "normal":
                df[row['feature']] = df[row['feature']].fillna(df[row['feature']].mean())
            elif row['nature'] == "category":
                df[row['feature']] = df[row['feature']].fillna(df[row['feature']].mode().values[0])
        
        return df



    def model_impute(self, dataset : pd.DataFrame, * , max_missing_prcnt : float = 75) -> pd.DataFrame:

        df = dataset.copy()

        summary = get_quick_summary(df,classify=True)
        high_missing_feats = summary.loc[summary['missing_percentage']>max_missing_prcnt,'feature']

        if high_missing_feats.size:
            df = df.drop(high_missing_feats,axis=1)
            summary = get_quick_summary(df,classify=True)
        
        categorical_features = summary.loc[summary['nature']=='category','feature'].to_list()
        if categorical_features:
            oe = OrdinalEncoder(handle_unknown='use_encoded_value',unknown_value=np.nan)
            df[categorical_features] = oe.fit_transform(df[categorical_features])
            self._encoder = oe

        missing_feats= summary.loc[(summary['missing_count']>0),['feature','nature']]
        stats_filled_data = self.stats_impute(df,summary)
        
        imputed_dataset = stats_filled_data.copy()
        
        for _ in  range(self._iterations):

            for _, row in missing_feats.iterrows():

                feat = row['feature']
                nature = row['nature']

                original_null_mask = df[feat].isna()
                not_null_indexes = df.index[~original_null_mask]
                null_indexes = df.index[original_null_mask]

                if null_indexes.empty: continue

                temp = imputed_dataset.copy()

                X_train = temp.loc[not_null_indexes].drop(columns=[feat])
                Y_train = temp.loc[not_null_indexes][feat]
                
                X_test = temp.loc[null_indexes].drop(columns=[feat])
                
                model = self._regressor if nature == "numeric" else self._classifier
                try:
                    if Y_train.empty or Y_train.nunique() <= 1: 
                        continue
                    model.fit(X_train, Y_train)
                    Y_test_imputed = model.predict(X_test)
                    
                    imputed_dataset.loc[null_indexes, feat] = Y_test_imputed
                except Exception as e:
                    print(f"Warning: Model failed for {feat}. Falling back to simple imputation. Error: {e}")
                    pass

        
        return imputed_dataset
    


