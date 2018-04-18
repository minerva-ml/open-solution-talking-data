import category_encoders as ce
import pandas as pd
from sklearn.externals import joblib

from steps.base import BaseTransformer
from steps.utils import get_logger

logger = get_logger()


class FeatureDispatcher(BaseTransformer):
    def __init__(self, numerical_columns, categorical_columns, timestamp_columns, return_df=False):
        self.numerical_columns = numerical_columns
        self.categorical_columns = categorical_columns
        self.timestamp_columns = timestamp_columns
        self.return_df = return_df

    def transform(self, X, y=None, **kwargs):
        outputs = {}

        if self.return_df:
            if self.numerical_columns is not None:
                outputs['numerical_features'] = X[self.numerical_columns]

            if self.categorical_columns is not None:
                outputs['categorical_features'] = X[self.categorical_columns]

            if self.timestamp_columns is not None:
                outputs['timestamp_features'] = X[self.timestamp_columns]
        else:
            if self.numerical_columns is not None:
                outputs['numerical_features'] = X[self.numerical_columns].values
                outputs['numerical_feature_names'] = self.numerical_columns

            if self.categorical_columns is not None:
                outputs['categorical_features'] = X[self.categorical_columns].values
                outputs['categorical_feature_names'] = self.categorical_columns

            if self.timestamp_columns is not None:
                outputs['timestamp_features'] = X[self.timestamp_columns].values
                outputs['timestamp_feature_names'] = self.timestamp_columns

        return outputs


class FeatureJoiner(BaseTransformer):
    def __init__(self, return_df=True):
        self.return_df = return_df

    def transform(self, numerical_feature_list, categorical_feature_list, **kwargs):
        outputs = {}

        if self.return_df:
            outputs['X'] = pd.concat(numerical_feature_list + categorical_feature_list, axis=1)
            outputs['feature_names'] = self._get_feature_names(numerical_feature_list + categorical_feature_list)
            outputs['categorical_features'] = self._get_feature_names(categorical_feature_list)
        else:
            raise NotImplementedError('only return_df=True is supported')
        return outputs

    def _get_feature_names(self, dataframes):
        feature_names = []
        for dataframe in dataframes:
            feature_names.extend(list(dataframe.columns))
        return feature_names


class BasicCategoricalEncoder(BaseTransformer):
    def __init__(self, **kwargs):
        self.params = kwargs
        self.encoder_class = None

    def fit(self, X, y, **kwargs):
        categorical_columns = list(X.columns)
        self.target_encoder = self.encoder_class(cols=categorical_columns, **self.params)
        self.target_encoder.fit(X, y)
        return self

    def transform(self, X, y=None, **kwargs):
        X_ = self.target_encoder.transform(X)
        return {'X': X_}

    def load(self, filepath):
        self.target_encoder = joblib.load(filepath)
        return self

    def save(self, filepath):
        joblib.dump(self.target_encoder, filepath)


class TargetEncoder(BasicCategoricalEncoder):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.encoder_class = ce.TargetEncoder


class BinaryEncoder(BasicCategoricalEncoder):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.encoder_class = ce.binary.BinaryEncoder
