"""Sparse, semantically aligned model pipelines for ACSIncome temporal transfer."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy import sparse
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from .semantic import SemanticMap, semantic_map_from_preprocessor

FEATURES=['AGEP','COW','SCHL','MAR','OCCP','POBP','RELP_HARM','WKHP','SEX','RAC1P']
NUMERIC=['AGEP','WKHP']
CATEGORICAL=['COW','SCHL','MAR','OCCP','POBP','RELP_HARM','SEX','RAC1P']

@dataclass
class ACSFittedModel:
    model_name:str
    preprocessor:ColumnTransformer
    estimator:object
    semantic_map:SemanticMap
    encoded_center:np.ndarray
    def transform(self,x:pd.DataFrame):
        return self.preprocessor.transform(x[FEATURES])
    def transform_dense(self,x:pd.DataFrame)->np.ndarray:
        z=self.transform(x)
        return z.toarray() if sparse.issparse(z) else np.asarray(z,float)
    def predict_proba(self,x:pd.DataFrame)->np.ndarray:
        z=self.transform(x)
        return np.asarray(self.estimator.predict_proba(z)[:, 1], dtype=float)
    def raw_score(self,x:pd.DataFrame)->np.ndarray:
        z=self.transform(x)
        if self.model_name=='lightgbm': return np.asarray(self.estimator.booster_.predict(z,raw_score=True),float)
        return np.asarray(self.estimator.decision_function(z),float)
    def explain(self,x:pd.DataFrame):
        z=self.transform_dense(x)
        if self.model_name=='logistic':
            coef=np.asarray(self.estimator.coef_[0],float)
            encoded=(z-self.encoded_center[None,:])*coef[None,:]
            base=float(self.estimator.intercept_[0]+self.encoded_center@coef)
        else:
            contrib=np.asarray(self.estimator.booster_.predict(z,pred_contrib=True),float)
            if contrib.shape[1]!=z.shape[1]+1: raise ValueError(f'Unexpected contribution shape {contrib.shape}')
            encoded=contrib[:,:-1]; bases=contrib[:,-1]
            if not np.allclose(bases,bases[0],atol=1e-10,rtol=1e-10): raise AssertionError('Nonconstant LightGBM base')
            base=float(bases[0])
        semantic=self.semantic_map.aggregate(encoded)
        return encoded,semantic,base

def build_preprocessor()->ColumnTransformer:
    num=Pipeline([('impute',SimpleImputer(strategy='median')),('scale',StandardScaler())])
    cat=Pipeline([('impute',SimpleImputer(strategy='most_frequent')),('onehot',OneHotEncoder(handle_unknown='ignore',sparse_output=True,dtype=np.float64))])
    return ColumnTransformer([('num',num,NUMERIC),('cat',cat,CATEGORICAL)],remainder='drop',sparse_threshold=1.0,verbose_feature_names_out=True)

def fit_acs_model(model_name:str,train:pd.DataFrame,y:np.ndarray,seed:int)->ACSFittedModel:
    prep=build_preprocessor(); z=prep.fit_transform(train[FEATURES])
    if model_name=='logistic':
        est=LogisticRegression(C=1.0,max_iter=800,solver='liblinear',random_state=seed)
    elif model_name=='lightgbm':
        est=LGBMClassifier(objective='binary',n_estimators=240,learning_rate=0.035,num_leaves=31,min_child_samples=40,subsample=1.0,colsample_bytree=1.0,reg_lambda=1.0,random_state=seed,n_jobs=1,verbosity=-1,deterministic=True,force_col_wise=True)
    else: raise ValueError(model_name)
    est.fit(z,np.asarray(y,int))
    center=np.asarray(z.mean(axis=0)).ravel()
    sm=semantic_map_from_preprocessor(prep,NUMERIC,CATEGORICAL)
    return ACSFittedModel(model_name,prep,est,sm,center)
