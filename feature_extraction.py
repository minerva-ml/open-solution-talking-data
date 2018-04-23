import category_encoders as ce
import pandas as pd
import numpy as np
from sklearn.externals import joblib

from steps.base import BaseTransformer
from steps.utils import get_logger

logger = get_logger()


class DataFrameByTypeSplitter(BaseTransformer):
    def __init__(self, numerical_columns, categorical_columns, timestamp_columns):
        self.numerical_columns = numerical_columns
        self.categorical_columns = categorical_columns
        self.timestamp_columns = timestamp_columns

    def transform(self, X, y=None, **kwargs):
        outputs = {}

        if self.numerical_columns is not None:
            outputs['numerical_features'] = X[self.numerical_columns]

        if self.categorical_columns is not None:
            outputs['categorical_features'] = X[self.categorical_columns]

        if self.timestamp_columns is not None:
            outputs['timestamp_features'] = X[self.timestamp_columns]

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


class ConfidenceRate(BaseTransformer):

    def __init__(self, confidence_level=100, categories=[['channel']]):
        self.confidence_level = confidence_level
        self.categories = categories

    def transform(self, X, y):
        concatenated_dataframe = pd.concat([X, y])
        new_features = []

        for category in self.categories:
            new_feature = '_'.join(category) + '_confidence_rate'
            new_features.append(new_feature)

            group_object = concatenated_dataframe.groupby(category)

            concatenated_dataframe = concatenated_dataframe.merge(
                group_object['is_attributed'].apply(self._rate_calculation).reset_index().rename(
                    index=str,
                    columns={'is_attributed': new_feature}
                )[category + [new_feature]],
                on=category, how='left'
            )

        return {'numerical_features': concatenated_dataframe[new_features]}

    def _rate_calculation(self, x):
        rate = x.sum() / float(x.count())
        confidence = np.min([1, np.log(x.count()) / np.log(self.confidence_level)])

        return rate * confidence * 100


class ConfidenceRateCutOff(BaseTransformer):

    def __init__(self, confidence_level=100, categories=[['channel']]):

        self.confidence_level = confidence_level
        self.categories = categories

    def transform(self, X, y):

        concatenated_dataframe = pd.concat([X, y])
        new_features = []

        for category in self.categories:
            new_feature = '_'.join(category) + '_confidence'
            new_features.append(new_feature)
            new_features.append(new_feature + '_rate')

            group_object = concatenated_dataframe.groupby(category)

            confidence = group_object['is_attributed'].apply(self._confidence_calculation).reset_index().rename(
                index=str,
                columns={'is_attributed': new_feature}
            )[category + [new_feature]]

            rate = group_object['is_attributed'].apply(self._rate_calculation).reset_index().rename(
                index=str,
                columns={'is_attributed': new_feature + '_rate'}
            )[category + [new_feature + '_rate']]

            new_columns = confidence.merge(rate, on=category)

            concatenated_dataframe = concatenated_dataframe.merge(new_columns, on=category, how='left')

        return {'numerical_features': concatenated_dataframe[new_features]}

    def _confidence_calculation(self, x):

        if x.count() > self.confidence_level:
            is_above_level = 1
        else:
            is_above_level = 0

        return is_above_level

    def _rate_calculation(self, x):

        if x.count() > self.confidence_level:
            rate = x.sum() / float(x.count())
        else:
            rate = 0

        return rate * 100
