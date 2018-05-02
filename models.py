import numpy as np
import lightgbm as lgb

from steps.misc import LightGBM


class LightGBMLowMemory(LightGBM):
    def fit(self, X, y, X_valid, y_valid, feature_names=None, categorical_features=None, **kwargs):
        X = X[feature_names].values.astype(np.float32)
        y = y.astype(np.int8)

        X_valid = X_valid[feature_names].values.astype(np.float32)
        y_valid = y_valid.astype(np.int8)

        train = lgb.Dataset(X, label=y)
        valid = lgb.Dataset(X_valid, label=y_valid)

        evaluation_results = {}
        self.estimator = lgb.train(self.model_config,
                                   train, valid_sets=[train, valid], valid_names=['train', 'valid'],
                                   evals_result=evaluation_results,
                                   num_boost_round=self.training_config.number_boosting_rounds,
                                   early_stopping_rounds=self.training_config.early_stopping_rounds,
                                   verbose_eval=self.model_config.verbose,
                                   feval=self.evaluation_function)
        return self
