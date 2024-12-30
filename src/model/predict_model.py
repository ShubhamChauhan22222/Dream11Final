import pandas as pd
import pickle
from datetime import datetime
import numpy as np
import lightgbm
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from math import floor
from sklearn.base import BaseEstimator, RegressorMixin
import sys
sys.path.append('/Users/greenkedia/Desktop/Dream11Final/src/')
class HybridModel(BaseEstimator, RegressorMixin):
    def __init__(self,
                 lgbm_params=None,
                 xgb_params=None,
                 xgb_weight=0.6):
        self.lgbm_params = lgbm_params if lgbm_params is not None else {
            'max_depth': 6,
            'num_leaves': 31,
            'min_child_samples': 20,
            'learning_rate': 0.05,
            'lambda_l1': 1.0,
            'lambda_l2': 1.0
        }
        self.xgb_params = xgb_params if xgb_params is not None else {
            'max_depth': 6,
            'learning_rate': 0.05,
            'min_child_weight': 1,
            'alpha': 1.0,
            'lambda': 1.0,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            "enable_categorical": True
        }
        self.xgb_weight = xgb_weight
        self.model1 = LGBMRegressor(**self.lgbm_params)
        self.model2 = XGBRegressor(**self.xgb_params)


    def fit(self, X, y):
        self.model1.fit(X, y)
        self.model2.fit(X, y)
        return self

    def predict(self, X):
        y_pred1 = self.model1.predict(X)
        y_pred2 = self.model2.predict(X)
        y_pred = y_pred1 + self.xgb_weight * y_pred2
        return y_pred



def predict(X_test,model):
    return model.predict(X_test)