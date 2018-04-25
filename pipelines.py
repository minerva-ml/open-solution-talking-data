from functools import partial

from sklearn.metrics import roc_auc_score

import feature_extraction as fe
from hyperparameter_tuning import RandomSearchOptimizer, NeptuneMonitor, SaveResults
from steps.adapters import to_numpy_label_inputs, identity_inputs
from steps.base import Step, Dummy
from steps.misc import LightGBM


def baseline(config, train_mode):
    if train_mode:
        features, features_valid = feature_extraction_v0(config, train_mode)
        light_gbm = classifier_lgbm((features, features_valid), config, train_mode)
    else:
        features = feature_extraction_v0(config, train_mode)
        light_gbm = classifier_lgbm(features, config, train_mode)

    output = Step(name='output',
                  transformer=Dummy(),
                  input_steps=[light_gbm],
                  adapter={'y_pred': ([(light_gbm.name, 'prediction')]),
                           },
                  cache_dirpath=config.env.cache_dirpath)
    return output


def solution_1(config, train_mode):
    if train_mode:
        features, features_valid = feature_extraction_v1(config, train_mode,
                                                         save_output=True, cache_output=True, load_saved_output=False)
        light_gbm = classifier_lgbm((features, features_valid), config, train_mode)
    else:
        features = feature_extraction_v1(config, train_mode, cache_output=True)
        light_gbm = classifier_lgbm(features, config, train_mode)

    output = Step(name='output',
                  transformer=Dummy(),
                  input_steps=[light_gbm],
                  adapter={'y_pred': ([(light_gbm.name, 'prediction')]),
                           },
                  cache_dirpath=config.env.cache_dirpath)
    return output


def feature_extraction_v0(config, train_mode, **kwargs):
    if train_mode:
        feature_by_type_split, feature_by_type_split_valid = _feature_by_type_splits(config, train_mode)
        categorical_features = Step(name='categorical_features',
                                    transformer=Dummy(),
                                    input_steps=[feature_by_type_split],
                                    adapter={'X': ([(feature_by_type_split.name, 'categorical_features')]),
                                             },
                                    cache_dirpath=config.env.cache_dirpath,
                                    **kwargs)
        categorical_features_valid = Step(name='categorical_features_valid',
                                          transformer=Dummy(),
                                          input_steps=[feature_by_type_split_valid],
                                          adapter={'X': ([(feature_by_type_split_valid.name, 'categorical_features')]),
                                                   },
                                          cache_dirpath=config.env.cache_dirpath,
                                          **kwargs)
        feature_combiner = _join_features(numerical_features=[],
                                          numerical_features_valid=[],
                                          categorical_features=[categorical_features],
                                          categorical_features_valid=[categorical_features_valid],
                                          config=config, train_mode=train_mode)
        return feature_combiner
    else:
        feature_by_type_split = _feature_by_type_splits(config, train_mode)
        categorical_features = Step(name='categorical_features',
                                    transformer=Dummy(),
                                    input_steps=[feature_by_type_split],
                                    adapter={'X': ([(feature_by_type_split.name, 'categorical_features')]),
                                             },
                                    cache_dirpath=config.env.cache_dirpath,
                                    **kwargs)
        feature_combiner = _join_features(numerical_features=[],
                                          numerical_features_valid=[],
                                          categorical_features=[categorical_features],
                                          categorical_features_valid=[],
                                          config=config, train_mode=train_mode)
        return feature_combiner


def feature_extraction_v1(config, train_mode, **kwargs):
    if train_mode:
        feature_by_type_split, feature_by_type_split_valid = _feature_by_type_splits(config, train_mode)
        time_delta, time_delta_valid = _time_deltas((feature_by_type_split, feature_by_type_split_valid),
                                                    config, train_mode, **kwargs)
        confidence_rate, confidence_rate_valid = _confidence_rates((feature_by_type_split, feature_by_type_split_valid),
                                                                   config, train_mode, **kwargs)

        feature_combiner, feature_combiner_valid = _join_features(numerical_features=[time_delta, confidence_rate],
                                                                  numerical_features_valid=[time_delta_valid,
                                                                                            confidence_rate_valid],
                                                                  categorical_features=[time_delta],
                                                                  categorical_features_valid=[time_delta_valid],
                                                                  config=config, train_mode=train_mode,
                                                                  **kwargs)
        return feature_combiner, feature_combiner_valid
    else:
        feature_by_type_split = _feature_by_type_splits(config, train_mode)
        time_delta = _time_deltas(feature_by_type_split, config, train_mode, **kwargs)
        confidence_rate = _confidence_rates(feature_by_type_split, config, train_mode, **kwargs)

        feature_combiner = _join_features(numerical_features=[time_delta, confidence_rate],
                                          numerical_features_valid=[],
                                          categorical_features=[time_delta],
                                          categorical_features_valid=[],
                                          config=config, train_mode=train_mode,
                                          **kwargs)
        return feature_combiner


def classifier_lgbm(features, config, train_mode, **kwargs):
    if train_mode:
        features_train, features_valid = features
        if config.random_search.light_gbm.n_runs:
            transformer = RandomSearchOptimizer(LightGBM, config.light_gbm,
                                                train_input_keys=[],
                                                valid_input_keys=['X_valid', 'y_valid'],
                                                score_func=roc_auc_score,
                                                maximize=True,
                                                n_runs=config.random_search.light_gbm.n_runs,
                                                callbacks=[NeptuneMonitor(
                                                    **config.random_search.light_gbm.callbacks.neptune_monitor),
                                                    SaveResults(
                                                        **config.random_search.light_gbm.callbacks.save_results),
                                                ]
                                                )
        else:
            transformer = LightGBM(**config.light_gbm)

        light_gbm = Step(name='light_gbm',
                         transformer=transformer,
                         input_data=['input'],
                         input_steps=[features_train, features_valid],
                         adapter={'X': ([(features_train.name, 'features')]),
                                  'y': ([('input', 'y')], to_numpy_label_inputs),
                                  'feature_names': ([(features_train.name, 'feature_names')]),
                                  'categorical_features': ([(features_train.name, 'categorical_features')]),
                                  'X_valid': ([(features_valid.name, 'features')]),
                                  'y_valid': ([('input', 'y_valid')], to_numpy_label_inputs),
                                  },
                         cache_dirpath=config.env.cache_dirpath,
                         **kwargs)
    else:
        light_gbm = Step(name='light_gbm',
                         transformer=LightGBM(**config.light_gbm),
                         input_steps=[features],
                         adapter={'X': ([(features.name, 'features')]),
                                  },
                         cache_dirpath=config.env.cache_dirpath,
                         **kwargs)
    return light_gbm


def _feature_by_type_splits(config, train_mode):
    if train_mode:
        feature_by_type_split = Step(name='feature_by_type_split',
                                     transformer=fe.DataFrameByTypeSplitter(**config.dataframe_by_type_splitter),
                                     input_data=['input'],
                                     adapter={'X': ([('input', 'X')]),
                                              },
                                     cache_dirpath=config.env.cache_dirpath)

        feature_by_type_split_valid = Step(name='feature_by_type_split_valid',
                                           transformer=feature_by_type_split,
                                           input_data=['input'],
                                           adapter={'X': ([('input', 'X_valid')]),
                                                    },
                                           cache_dirpath=config.env.cache_dirpath)

        return feature_by_type_split, feature_by_type_split_valid

    else:
        feature_by_type_split = Step(name='feature_by_type_split',
                                     transformer=fe.DataFrameByTypeSplitter(**config.dataframe_by_type_splitter),
                                     input_data=['input'],
                                     adapter={'X': ([('input', 'X')]),
                                              },
                                     cache_dirpath=config.env.cache_dirpath)

        return feature_by_type_split


def _categorical_frequency_filters(dispatchers, config, train_mode, **kwargs):
    if train_mode:
        feature_by_type_split, feature_by_type_split_valid = dispatchers
        categorical_filter = Step(name='categorical_filter',
                                  transformer=fe.CategoricalFilter(**config.categorical_filter),
                                  input_steps=[feature_by_type_split],
                                  adapter={
                                      'categorical_features': ([(feature_by_type_split.name, 'categorical_features')]),
                                  },
                                  cache_dirpath=config.env.cache_dirpath,
                                  **kwargs)

        categorical_filter_valid = Step(name='categorical_filter_valid',
                                        transformer=categorical_filter,
                                        input_steps=[feature_by_type_split_valid],
                                        adapter={'categorical_features': (
                                            [(feature_by_type_split_valid.name, 'categorical_features')]),
                                        },
                                        cache_dirpath=config.env.cache_dirpath,
                                        **kwargs)

        return categorical_filter, categorical_filter_valid

    else:
        feature_by_type_split = dispatchers

        categorical_filter = Step(name='categorical_filter',
                                  transformer=fe.CategoricalFilter(**config.categorical_filter),
                                  input_data=['input'],
                                  input_steps=[feature_by_type_split],
                                  adapter={
                                      'categorical_features': ([(feature_by_type_split.name, 'categorical_features')]),
                                  },
                                  cache_dirpath=config.env.cache_dirpath,
                                  **kwargs)

        return categorical_filter


def _target_encoders(dispatchers, config, train_mode, **kwargs):
    if train_mode:
        feature_by_type_split, feature_by_type_split_valid = dispatchers
        target_encoder = Step(name='target_encoder',
                              transformer=fe.TargetEncoder(**config.target_encoder),
                              input_data=['input'],
                              input_steps=[feature_by_type_split],
                              adapter={'X': ([(feature_by_type_split.name, 'categorical_features')]),
                                       'y': ([('input', 'y')], to_numpy_label_inputs),
                                       },
                              cache_dirpath=config.env.cache_dirpath,
                              **kwargs)

        target_encoder_valid = Step(name='target_encoder_valid',
                                    transformer=target_encoder,
                                    input_data=['input'],
                                    input_steps=[feature_by_type_split_valid],
                                    adapter={'X': ([(feature_by_type_split_valid.name, 'categorical_features')]),
                                             'y': ([('input', 'y_valid')], to_numpy_label_inputs),
                                             },
                                    cache_dirpath=config.env.cache_dirpath,
                                    **kwargs)

        return target_encoder, target_encoder_valid

    else:
        feature_by_type_split = dispatchers

        target_encoder = Step(name='target_encoder',
                              transformer=fe.TargetEncoder(**config.target_encoder),
                              input_data=['input'],
                              input_steps=[feature_by_type_split],
                              adapter={'X': ([(feature_by_type_split.name, 'categorical_features')]),
                                       },
                              cache_dirpath=config.env.cache_dirpath,
                              **kwargs)

        return target_encoder


def _binary_encoders(dispatchers, config, train_mode, **kwargs):
    if train_mode:
        feature_by_type_split, feature_by_type_split_valid = dispatchers
        binary_encoder = Step(name='binary_encoder',
                              transformer=fe.BinaryEncoder(),
                              input_steps=[feature_by_type_split],
                              adapter={'X': ([(feature_by_type_split.name, 'categorical_features')]),
                                       },
                              cache_dirpath=config.env.cache_dirpath,
                              **kwargs)

        binary_encoder_valid = Step(name='binary_encoder_valid',
                                    transformer=binary_encoder,
                                    input_steps=[feature_by_type_split_valid],
                                    adapter={'X': ([(feature_by_type_split_valid.name, 'categorical_features')]),
                                             },
                                    cache_dirpath=config.env.cache_dirpath,
                                    **kwargs)

        return binary_encoder, binary_encoder_valid

    else:
        feature_by_type_split = dispatchers

        binary_encoder = Step(name='binary_encoder',
                              transformer=fe.BinaryEncoder(),
                              input_steps=[feature_by_type_split],
                              adapter={'X': ([(feature_by_type_split.name, 'categorical_features')]),
                                       },
                              cache_dirpath=config.env.cache_dirpath,
                              **kwargs)

        return binary_encoder


def _time_deltas(dispatchers, config, train_mode, **kwargs):
    if train_mode:
        feature_by_type_split, feature_by_type_split_valid = dispatchers
        time_delta = Step(name='time_delta',
                          transformer=fe.TimeDelta(**config.time_delta),
                          input_steps=[feature_by_type_split],
                          adapter={'categorical_features': ([(feature_by_type_split.name, 'categorical_features')]),
                                   'timestamp_features': ([(feature_by_type_split.name, 'timestamp_features')])
                                   },
                          cache_dirpath=config.env.cache_dirpath,
                          **kwargs)

        time_delta_valid = Step(name='time_delta_valid',
                                transformer=time_delta,
                                input_steps=[feature_by_type_split_valid],
                                adapter={'categorical_features': (
                                    [(feature_by_type_split_valid.name, 'categorical_features')]),
                                    'timestamp_features': (
                                        [(feature_by_type_split_valid.name, 'timestamp_features')])
                                },
                                cache_dirpath=config.env.cache_dirpath,
                                **kwargs)

        return time_delta, time_delta_valid

    else:
        feature_by_type_split = dispatchers
        time_delta = Step(name='time_delta',
                          transformer=fe.TimeDelta(**config.time_delta),
                          input_steps=[feature_by_type_split],
                          adapter={'categorical_features': ([(feature_by_type_split.name, 'categorical_features')]),
                                   'timestamp_features': ([(feature_by_type_split.name, 'timestamp_features')])
                                   },
                          cache_dirpath=config.env.cache_dirpath,
                          **kwargs)

        return time_delta


def _confidence_rates(dispatchers, config, train_mode, **kwargs):
    if train_mode:
        feature_by_type_split, feature_by_type_split_valid = dispatchers
        confidence_rates = Step(name='confidence_rates',
                                transformer=fe.ConfidenceRate(**config.confidence_rate),
                                input_data=['input'],
                                input_steps=[feature_by_type_split],
                                adapter={
                                    'categorical_features': ([(feature_by_type_split.name, 'categorical_features')]),
                                    'target': ([('input', 'y')])
                                },
                                cache_dirpath=config.env.cache_dirpath,
                                **kwargs)

        confidence_rates_valid = Step(name='confidence_rates_valid',
                                      transformer=confidence_rates,
                                      input_data=['input'],
                                      input_steps=[feature_by_type_split_valid],
                                      adapter={'categorical_features': (
                                          [(feature_by_type_split_valid.name, 'categorical_features')]),
                                          'target': ([('input', 'y_valid')])
                                      },
                                      cache_dirpath=config.env.cache_dirpath,
                                      **kwargs)

        return confidence_rates, confidence_rates_valid

    else:
        feature_by_type_split = dispatchers
        confidence_rates = Step(name='confidence_rates',
                                transformer=fe.ConfidenceRate(**config.confidence_rate),
                                input_data=['input'],
                                input_steps=[feature_by_type_split],
                                adapter={
                                    'categorical_features': ([(feature_by_type_split.name, 'categorical_features')]),
                                    'target': ([('input', 'y')])
                                },
                                cache_dirpath=config.env.cache_dirpath,
                                **kwargs)

        return confidence_rates


def _join_features(numerical_features, numerical_features_valid,
                   categorical_features, categorical_features_valid,
                   config, train_mode,
                   **kwargs):
    if train_mode:
        feature_joiner = Step(name='feature_joiner',
                              transformer=fe.FeatureJoiner(),
                              input_steps=numerical_features + categorical_features,
                              adapter={
                                  'numerical_feature_list': (
                                      [(feature.name, 'numerical_features') for feature in numerical_features],
                                      identity_inputs),
                                  'categorical_feature_list': (
                                      [(feature.name, 'categorical_features') for feature in categorical_features],
                                      identity_inputs),
                              },
                              cache_dirpath=config.env.cache_dirpath,
                              **kwargs)

        feature_joiner_valid = Step(name='feature_joiner_valid',
                                    transformer=feature_joiner,
                                    input_steps=numerical_features_valid + categorical_features_valid,
                                    adapter={'numerical_feature_list': (
                                        [(feature.name, 'numerical_features') for feature in numerical_features_valid],
                                        identity_inputs),
                                        'categorical_feature_list': (
                                            [(feature.name, 'categorical_features') for feature in
                                             categorical_features_valid],
                                            identity_inputs),
                                    },
                                    cache_dirpath=config.env.cache_dirpath,
                                    **kwargs)

        return feature_joiner, feature_joiner_valid

    else:
        feature_joiner = Step(name='feature_joiner',
                              transformer=fe.FeatureJoiner(),
                              input_steps=numerical_features + categorical_features,
                              adapter={
                                  'numerical_feature_list': (
                                      [(feature.name, 'numerical_features') for feature in numerical_features],
                                      identity_inputs),
                                  'categorical_feature_list': (
                                      [(feature.name, 'categorical_features') for feature in categorical_features],
                                      identity_inputs),
                              },
                              cache_dirpath=config.env.cache_dirpath,
                              **kwargs)

        return feature_joiner


PIPELINES = {'baseline': {'train': partial(baseline, train_mode=True),
                          'inference': partial(baseline, train_mode=False)},
             'solution_1': {'train': partial(solution_1, train_mode=True),
                            'inference': partial(solution_1, train_mode=False)},
             }
