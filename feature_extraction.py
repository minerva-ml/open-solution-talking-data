import category_encoders as ce
import numpy as np
import pandas as pd
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
    def transform(self, numerical_feature_list, categorical_feature_list, **kwargs):
        features = numerical_feature_list + categorical_feature_list
        for feature in features:
            feature.reset_index(drop=True, inplace=True)

        outputs = {}
        outputs['features'] = pd.concat(features, axis=1)
        outputs['feature_names'] = self._get_feature_names(features)
        outputs['categorical_features'] = self._get_feature_names(categorical_feature_list)
        return outputs

    def _get_feature_names(self, dataframes):
        feature_names = []
        for dataframe in dataframes:
            feature_names.extend(list(dataframe.columns))
        return feature_names


class CategoricalFilter(BaseTransformer):
    def __init__(self, categorical_columns, min_frequencies, impute_value=np.nan):
        self.categorical_columns = categorical_columns
        self.min_frequencies = min_frequencies
        self.impute_value = impute_value
        self.category_levels_to_remove = {}

    def fit(self, categorical_features):
        for column, threshold in zip(self.categorical_columns, self.min_frequencies):
            value_counts = categorical_features[column].value_counts()
            self.category_levels_to_remove[column] = value_counts[value_counts <= threshold].index.tolist()
        return self

    def transform(self, categorical_features):
        for column, levels_to_remove in self.category_levels_to_remove.items():
            if levels_to_remove:
                categorical_features[column].replace(levels_to_remove, self.impute_value, inplace=True)
            categorical_features['{}_infrequent'.format(column)] = (
                categorical_features[column] == self.impute_value).astype(int)
        return {'categorical_features': categorical_features}

    def load(self, filepath):
        params = joblib.load(filepath)
        self.categorical_columns = params['categorical_columns']
        self.min_frequencies = params['min_frequencies']
        self.impute_value = params['impute_value']
        self.category_levels_to_remove = params['category_levels_to_remove']
        return self

    def save(self, filepath):
        params = {}
        params['categorical_columns'] = self.categorical_columns
        params['min_frequencies'] = self.min_frequencies
        params['impute_value'] = self.impute_value
        params['category_levels_to_remove'] = self.category_levels_to_remove
        joblib.dump(params, filepath)


class TargetEncoder(BaseTransformer):
    def __init__(self, **kwargs):
        self.params = kwargs
        self.encoder_class = ce.TargetEncoder

    def fit(self, X, y, **kwargs):
        categorical_columns = list(X.columns)
        self.target_encoder = self.encoder_class(cols=categorical_columns, **self.params)
        self.target_encoder.fit(X, y)
        return self

    def transform(self, X, y=None, **kwargs):
        X_ = self.target_encoder.transform(X)
        return {'numerical_features': X_}

    def load(self, filepath):
        self.target_encoder = joblib.load(filepath)
        return self

    def save(self, filepath):
        joblib.dump(self.target_encoder, filepath)


class BinaryEncoder(BaseTransformer):
    def __init__(self, **kwargs):
        self.params = kwargs
        self.encoder_class = ce.binary.BinaryEncoder

    def fit(self, X, **kwargs):
        categorical_columns = list(X.columns)
        self.binary_encoder = self.encoder_class(cols=categorical_columns, **self.params)
        self.binary_encoder.fit(X)
        return self

    def transform(self, X, **kwargs):
        X_ = self.binary_encoder.transform(X)
        return {'numerical_features': X_}

    def load(self, filepath):
        self.target_encoder = joblib.load(filepath)
        return self

    def save(self, filepath):
        joblib.dump(self.target_encoder, filepath)


class TimeDelta(BaseTransformer):
    def __init__(self, groupby_specs, timestamp_column):
        self.groupby_specs = groupby_specs
        self.timestamp_column = timestamp_column

    @property
    def time_delta_names(self):
        time_delta_names = ['{}_time_delta'.format('_'.join(groupby_spec))
                            for groupby_spec in self.groupby_specs]
        return time_delta_names

    @property
    def is_null_names(self):
        is_null_names = ['{}_is_nan'.format('_'.join(groupby_spec))
                         for groupby_spec in self.groupby_specs]
        return is_null_names

    def transform(self, categorical_features, timestamp_features):
        X = pd.concat([categorical_features, timestamp_features], axis=1)
        for groupby_spec, time_delta_name, is_null_name in zip(self.groupby_specs,
                                                               self.time_delta_names,
                                                               self.is_null_names):
            X[time_delta_name] = X.groupby(groupby_spec)[self.timestamp_column].apply(self._time_delta).reset_index(
                level=list(range(len(groupby_spec))), drop=True)
            X[is_null_name] = pd.isnull(X[time_delta_name]).astype(int)
            X[time_delta_name].fillna(0, inplace=True)
        return {'numerical_features': X[self.time_delta_names],
                'categorical_features': X[self.is_null_names]}

    def _time_delta(self, groupby_object):
        if len(groupby_object) == 1:
            return pd.Series(np.nan, index=groupby_object.index)
        else:
            groupby_object = groupby_object.sort_values().diff().dt.seconds
            return groupby_object


class GroupbyCounts(BaseTransformer):

    def __init__(self, categories=[]):
        self.categories = categories

    def transform(self, X, y):
        new_features = []

        for category in self.categories:
            new_feature = '{}_counts'.format('_'.join(category))
            new_features.append(new_feature)

            group_object = X.groupby(category)

            X = X.merge(
                group_object.size().reset_index().rename(index=str, columns={0: new_feature})[category + [new_feature]],
                on=category, how='left'
                )

        return {'numerical_features': X[new_features]}


class ConfidenceRate(BaseTransformer):
    def __init__(self, confidence_level=100, categories=[]):
        self.confidence_level = confidence_level
        self.categories = categories

    def transform(self, categorical_features, target):
        concatenated_dataframe = pd.concat([categorical_features, target], axis=1)
        new_features = []

        for category in self.categories:
            new_feature = '{}_confidence_rate'.format('_'.join(category))
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
