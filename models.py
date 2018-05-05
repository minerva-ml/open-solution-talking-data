import gc
import os

from attrdict import AttrDict
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from sklearn.externals import joblib
from sklearn import ensemble
from sklearn.datasets import dump_svmlight_file

from steps.base import BaseTransformer
from steps.misc import LightGBM


class LightGBMLowMemory(LightGBM):
    def fit(self, X, y, X_valid, y_valid, feature_names=None, categorical_features=None, **kwargs):
        X = X[feature_names].values.astype(np.float32)
        X_valid = X_valid[feature_names].values.astype(np.float32)

        train = lgb.Dataset(X, label=y)
        valid = lgb.Dataset(X_valid, label=y_valid)

        evaluation_results = {}
        self.estimator = lgb.train(self.model_config,
#                                    train, valid_sets=[train, valid], valid_names=['train', 'valid'],
                                   train, valid_sets=[valid], valid_names=['valid'],
                                   evals_result=evaluation_results,
                                   num_boost_round=self.training_config.number_boosting_rounds,
                                   early_stopping_rounds=self.training_config.early_stopping_rounds,
                                   verbose_eval=self.model_config.verbose,
                                   feval=self.evaluation_function)
        return self


class XGBoost(BaseTransformer):
    def __init__(self, **params):
        self.params = params
        self.training_params = ['number_boosting_rounds', 'early_stopping_rounds', 'maximize', 'temp_dir']
        self.evaluation_results = {}
        self.evaluation_function = None

    @property
    def model_config(self):
        return AttrDict({param: value for param, value in self.params.items()
                         if param not in self.training_params})

    @property
    def training_config(self):
        return AttrDict({param: value for param, value in self.params.items()
                         if param in self.training_params})

    def fit(self, X, y, X_valid, y_valid, **kwargs):
        X = X.values.astype(np.float32)
        X_valid = X_valid.values.astype(np.float32)

        train = self._get_DMatrix(X, y, 'train')
        valid = self._get_DMatrix(X_valid, y_valid, 'valid')

        self.estimator = xgb.train(self.model_config,
                                   train,
                                   num_boost_round=self.training_config.number_boosting_rounds,
                                   evals=[(train, 'train'), (valid, 'valid')],
                                   evals_result=self.evaluation_results,
                                   maximize=self.training_config.maximize,
                                   early_stopping_rounds=self.training_config.early_stopping_rounds,
                                   verbose_eval=self.model_config.verbose,
                                   feval=self.evaluation_function)
        return self

    def transform(self, X, **kwargs):
        X = X.values.astype(np.float32)
        data = xgb.DMatrix(data=X)
        prediction = self.estimator.predict(data)
        return {'prediction': prediction}

    def load(self, filepath):
        d = joblib.load(filepath)
        self.estimator = d['estimator']
        self.evaluation_results = d['eval_results']
        return self

    def save(self, filepath):
        joblib.dump({'estimator': self.estimator,
                     'eval_results': self.evaluation_results}, filepath)

    def _get_DMatrix(self, X, y, suffix):
        temp_filepath = os.path.join(self.training_config['temp_dir'], '{}.libsvm'.format(suffix))
        dump_svmlight_file(X, y, temp_filepath)

        del X, y
        gc.collect()
        return xgb.DMatrix(temp_filepath)


class RandomForestClassifier(BaseTransformer):
    def __init__(self, **params):
        self.estimator = ensemble.RandomForestClassifier(**params)

    def fit(self, X, y, **kwargs):
        X = X.values.astype(np.float32)
        self.estimator.fit(X, y)
        return self

    def transform(self, X, **kwargs):
        X = X.values.astype(np.float32)
        prediction = self.estimator.predict_proba(X)[:, 1]
        return {'prediction': prediction}

    def load(self, filepath):
        self.estimator = joblib.load(filepath)
        return self

    def save(self, filepath):
        joblib.dump(self.estimator, filepath)
