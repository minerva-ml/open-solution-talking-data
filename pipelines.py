from functools import partial

from sklearn.metrics import roc_auc_score

import feature_extraction as fe
from hyperparameter_tuning import RandomSearchOptimizer, NeptuneMonitor, SaveResults
from steps.adapters import to_numpy_label_inputs, identity_inputs
from steps.base import Step, Dummy
from steps.preprocessing import Normalizer
from models import LightGBMLowMemory as LightGBM, XGBoost, LogisticRegression


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


def features_v1_lgbm(config, train_mode):
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


def features_v2_lgbm(config, train_mode):
    if train_mode:
        features, features_valid = feature_extraction_v2(config, train_mode,
                                                         save_output=True, cache_output=True, load_saved_output=True)
        light_gbm = classifier_lgbm((features, features_valid), config, train_mode)
    else:
        features = feature_extraction_v2(config, train_mode, cache_output=True)
        light_gbm = classifier_lgbm(features, config, train_mode)

    output = Step(name='output',
                  transformer=Dummy(),
                  input_steps=[light_gbm],
                  adapter={'y_pred': ([(light_gbm.name, 'prediction')]),
                           },
                  cache_dirpath=config.env.cache_dirpath)
    return output


def features_v2_xgboost(config, train_mode):
    if train_mode:
        features, features_valid = feature_extraction_v2(config, train_mode,
                                                         save_output=True, cache_output=True, load_saved_output=True)
        xgboost = classifier_xgboost((features, features_valid), config, train_mode)
    else:
        features = feature_extraction_v2(config, train_mode, cache_output=True)
        xgboost = classifier_xgboost(features, config, train_mode)

    output = Step(name='output',
                  transformer=Dummy(),
                  input_steps=[xgboost],
                  adapter={'y_pred': ([(xgboost.name, 'prediction')]),
                           },
                  cache_dirpath=config.env.cache_dirpath)
    return output


def features_v2_log_reg(config, train_mode):
    if train_mode:
        features, features_valid = feature_extraction_v2(config, train_mode,
                                                         save_output=True, cache_output=True, load_saved_output=True)

        features_normalized, features_normalized_valid = _normalize((features, features_valid), config, train_mode,
                                                                    save_output=True, cache_output=True,
                                                                    load_saved_output=True)

        log_reg = classifier_log_reg((features_normalized, features_normalized_valid), config, train_mode)
    else:
        features = feature_extraction_v2(config, train_mode, cache_output=True)
        features_normalized = _normalize(features, config, train_mode, cache_output=True)
        log_reg = classifier_log_reg(features_normalized, config, train_mode)

    output = Step(name='output',
                  transformer=Dummy(),
                  input_steps=[log_reg],
                  adapter={'y_pred': ([(log_reg.name, 'prediction')]),
                           },
                  cache_dirpath=config.env.cache_dirpath)
    return output


def feature_extraction_v0(config, train_mode, **kwargs):
    if train_mode:
        feature_by_type_split, feature_by_type_split_valid = _feature_by_type_splits(config, train_mode)
        categorical_features = Step(name='categorical_features',
                                    transformer=Dummy(),
                                    input_steps=[feature_by_type_split],
                                    adapter={'categorical_features': (
                                        [(feature_by_type_split.name, 'categorical_features')]),
                                    },
                                    cache_dirpath=config.env.cache_dirpath,
                                    **kwargs)
        categorical_features_valid = Step(name='categorical_features_valid',
                                          transformer=Dummy(),
                                          input_steps=[feature_by_type_split_valid],
                                          adapter={'categorical_features': (
                                              [(feature_by_type_split_valid.name, 'categorical_features')]),
                                          },
                                          cache_dirpath=config.env.cache_dirpath,
                                          **kwargs)
        feature_combiner = _join_features(numerical_features=[],
                                          numerical_features_valid=[],
                                          categorical_features=[categorical_features],
                                          categorical_features_valid=[categorical_features_valid],
                                          config=config, train_mode=train_mode, **kwargs)
        return feature_combiner
    else:
        feature_by_type_split = _feature_by_type_splits(config, train_mode)
        categorical_features = Step(name='categorical_features',
                                    transformer=Dummy(),
                                    input_steps=[feature_by_type_split],
                                    adapter={'categorical_features': (
                                        [(feature_by_type_split.name, 'categorical_features')]),
                                    },
                                    cache_dirpath=config.env.cache_dirpath,
                                    **kwargs)
        feature_combiner = _join_features(numerical_features=[],
                                          numerical_features_valid=[],
                                          categorical_features=[categorical_features],
                                          categorical_features_valid=[],
                                          config=config, train_mode=train_mode, **kwargs)
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
                                                                  categorical_features=[time_delta, confidence_rate],
                                                                  categorical_features_valid=[time_delta_valid,
                                                                                              confidence_rate_valid],
                                                                  config=config, train_mode=train_mode, **kwargs)
        return feature_combiner, feature_combiner_valid
    else:
        feature_by_type_split = _feature_by_type_splits(config, train_mode)
        time_delta = _time_deltas(feature_by_type_split, config, train_mode, **kwargs)
        confidence_rate = _confidence_rates(feature_by_type_split, config, train_mode, **kwargs)

        feature_combiner = _join_features(numerical_features=[time_delta, confidence_rate],
                                          numerical_features_valid=[],
                                          categorical_features=[time_delta, confidence_rate],
                                          categorical_features_valid=[],
                                          config=config, train_mode=train_mode, **kwargs)
        return feature_combiner


def feature_extraction_v2(config, train_mode, **kwargs):
    if train_mode:
        feature_by_type_split, feature_by_type_split_valid = _feature_by_type_splits(config, train_mode)
        time_delta, time_delta_valid = _time_deltas((feature_by_type_split, feature_by_type_split_valid),
                                                    config, train_mode, **kwargs)
        groupby_aggregation, groupby_aggregation_valid = _groupby_aggregations(
            (feature_by_type_split, feature_by_type_split_valid),
            config, train_mode, **kwargs)
        blacklist, blacklist_valid = _blacklists((feature_by_type_split, feature_by_type_split_valid),
                                                 config, train_mode, **kwargs)
        confidence_rate, confidence_rate_valid = _confidence_rates((feature_by_type_split, feature_by_type_split_valid),
                                                                   config, train_mode, **kwargs)

        target_encoder, target_encoder_valid = _target_encoders((feature_by_type_split, feature_by_type_split_valid),
                                                                config, train_mode, **kwargs)

        feature_combiner, feature_combiner_valid = _join_features(numerical_features=[time_delta,
                                                                                      groupby_aggregation,
                                                                                      confidence_rate,
                                                                                      target_encoder],
                                                                  numerical_features_valid=[time_delta_valid,
                                                                                            groupby_aggregation_valid,
                                                                                            confidence_rate_valid,
                                                                                            target_encoder_valid],
                                                                  categorical_features=[time_delta,
                                                                                        blacklist,
                                                                                        confidence_rate,
                                                                                        target_encoder],
                                                                  categorical_features_valid=[time_delta_valid,
                                                                                              blacklist_valid,
                                                                                              confidence_rate_valid,
                                                                                              target_encoder_valid],
                                                                  config=config, train_mode=train_mode, **kwargs)
        return feature_combiner, feature_combiner_valid
    else:
        feature_by_type_split = _feature_by_type_splits(config, train_mode)
        time_delta = _time_deltas(feature_by_type_split, config, train_mode, **kwargs)
        groupby_aggregation = _groupby_aggregations(feature_by_type_split, config, train_mode, **kwargs)
        blacklist = _blacklists(feature_by_type_split, config, train_mode, **kwargs)
        confidence_rate = _confidence_rates(feature_by_type_split, config, train_mode, **kwargs)
        target_encoder = _target_encoders(feature_by_type_split, config, train_mode, **kwargs)

        feature_combiner = _join_features(numerical_features=[time_delta, groupby_aggregation,
                                                              confidence_rate, target_encoder],
                                          numerical_features_valid=[],
                                          categorical_features=[time_delta, blacklist, confidence_rate, target_encoder],
                                          categorical_features_valid=[],
                                          config=config, train_mode=train_mode, **kwargs)
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


def classifier_xgboost(features, config, train_mode, **kwargs):
    if train_mode:
        features_train, features_valid = features
        if config.random_search.xgboost.n_runs:
            transformer = RandomSearchOptimizer(XGBoost, config.xgboost,
                                                train_input_keys=[],
                                                valid_input_keys=['X_valid', 'y_valid'],
                                                score_func=roc_auc_score,
                                                maximize=True,
                                                n_runs=config.random_search.xgboost.n_runs,
                                                callbacks=[NeptuneMonitor(
                                                    **config.random_search.xgboost.callbacks.neptune_monitor),
                                                    SaveResults(
                                                        **config.random_search.xgboost.callbacks.save_results),
                                                ]
                                                )
        else:
            transformer = XGBoost(**config.xgboost)

        xgboost = Step(name='xgboost',
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
                       cache_dirpath=config.env.cache_dirpath, **kwargs)
    else:
        xgboost = Step(name='xgboost',
                       transformer=XGBoost(**config.xgboost),
                       input_steps=[features],
                       adapter={'X': ([(features.name, 'features')]),
                                },
                       cache_dirpath=config.env.cache_dirpath, **kwargs)
    return xgboost


def classifier_log_reg(features, config, train_mode, **kwargs):
    if train_mode:
        features_train, features_valid = features

        if config.random_search.log_reg.n_runs:
            transformer = RandomSearchOptimizer(LogisticRegression, config.log_reg,
                                            train_input_keys=[],
                                            valid_input_keys=['X_valid', 'y_valid'],
                                            score_func=roc_auc_score,
                                            maximize=True,
                                            n_runs=config.random_search.log_reg.n_runs,
                                            callbacks=[NeptuneMonitor(
                                                **config.random_search.log_reg.callbacks.neptune_monitor),
                                                SaveResults(
                                                    **config.random_search.log_reg.callbacks.save_results),
                                            ]
                                            )
        else:
            transformer = LogisticRegression(**config.log_reg)

        log_reg = Step(name='log_reg',
                       transformer=transformer,
                       input_data=['input'],
                       input_steps=[features_train, features_valid],
                       adapter={'X': ([(features_train.name, 'X')]),
                                'y': ([('input', 'y')], to_numpy_label_inputs),
                                'X_valid': ([(features_valid.name, 'X')]),
                                'y_valid': ([('input', 'y_valid')], to_numpy_label_inputs),
                                },
                       cache_dirpath=config.env.cache_dirpath, **kwargs)
    else:

        log_reg = Step(name='log_reg',
                       transformer=LogisticRegression(**config.log_reg),
                       input_steps=[features],
                       adapter={'X': ([(features.name, 'features')]),
                                },
                       cache_dirpath=config.env.cache_dirpath, **kwargs)
    return log_reg


def _normalize(features, config, train_mode, **kwargs):
    if train_mode:
        feature_train, features_valid = features
        normalizer = Step(name='normalizer',
                          transformer=Normalizer(),
                          input_steps=[feature_train],
                          adapter={
                              'X': ([(feature_train.name, 'features')]),
                          },
                          cache_dirpath=config.env.cache_dirpath, **kwargs)

        normalizer_valid = Step(name='normalizer_valid',
                                transformer=normalizer,
                                input_steps=[features_valid],
                                adapter={'X': (
                                    [(features_valid.name, 'features')]),
                                },
                                cache_dirpath=config.env.cache_dirpath, **kwargs)

        return normalizer, normalizer_valid

    else:
        normalizer = Step(name='normalizer',
                          transformer=Normalizer(),
                          input_steps=[features],
                          adapter={
                              'X': ([(features.name, 'features')]),
                          },
                          cache_dirpath=config.env.cache_dirpath,
                          **kwargs)

        return normalizer


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
                              transformer=fe.TargetEncoderNSplits(**config.target_encoder),
                              input_data=['input'],
                              input_steps=[feature_by_type_split],
                              adapter={
                                  'categorical_features': ([(feature_by_type_split.name, 'categorical_features')]),
                                  'target': ([('input', 'y')])
                              },
                              cache_dirpath=config.env.cache_dirpath,
                              **kwargs)

        target_encoder_valid = Step(name='target_encoder_valid',
                                    transformer=target_encoder,
                                    input_data=['input'],
                                    input_steps=[feature_by_type_split_valid],
                                    adapter={'categorical_features': (
                                        [(feature_by_type_split_valid.name, 'categorical_features')]),
                                        'target': ([('input', 'y_valid')])
                                    },
                                    cache_dirpath=config.env.cache_dirpath,
                                    **kwargs)

        return target_encoder, target_encoder_valid

    else:
        feature_by_type_split = dispatchers
        target_encoder = Step(name='target_encoder',
                              transformer=fe.TargetEncoderNSplits(**config.target_encoder),
                              input_data=['input'],
                              input_steps=[feature_by_type_split],
                              adapter={
                                  'categorical_features': ([(feature_by_type_split.name, 'categorical_features')]),
                                  'target': ([('input', 'y')])
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


def _groupby_aggregations(dispatchers, config, train_mode, **kwargs):
    if train_mode:
        feature_by_type_split, feature_by_type_split_valid = dispatchers
        groupby_aggregations = Step(name='groupby_aggregations',
                                    transformer=fe.GroupbyAggregations(**config.groupby_aggregation),
                                    input_data=['input'],
                                    input_steps=[feature_by_type_split],
                                    adapter={
                                        'categorical_features': ([(feature_by_type_split.name, 'categorical_features')])
                                    },
                                    cache_dirpath=config.env.cache_dirpath,
                                    **kwargs)

        groupby_aggregations_valid = Step(name='groupby_aggregations_valid',
                                          transformer=groupby_aggregations,
                                          input_data=['input'],
                                          input_steps=[feature_by_type_split_valid],
                                          adapter={'categorical_features': (
                                              [(feature_by_type_split_valid.name, 'categorical_features')])
                                          },
                                          cache_dirpath=config.env.cache_dirpath,
                                          **kwargs)

        return groupby_aggregations, groupby_aggregations_valid

    else:
        feature_by_type_split = dispatchers
        groupby_aggregations = Step(name='groupby_aggregations',
                                    transformer=fe.GroupbyAggregations(**config.groupby_aggregation),
                                    input_data=['input'],
                                    input_steps=[feature_by_type_split],
                                    adapter={
                                        'categorical_features': ([(feature_by_type_split.name, 'categorical_features')])
                                    },
                                    cache_dirpath=config.env.cache_dirpath,
                                    **kwargs)

        return groupby_aggregations


def _blacklists(dispatchers, config, train_mode, **kwargs):
    if train_mode:
        feature_by_type_split, feature_by_type_split_valid = dispatchers
        blacklists = Step(name='blacklists',
                          transformer=fe.Blacklist(**config.blacklist),
                          input_data=['input'],
                          input_steps=[feature_by_type_split],
                          adapter={
                              'categorical_features': ([(feature_by_type_split.name, 'categorical_features')])
                          },
                          cache_dirpath=config.env.cache_dirpath,
                          **kwargs)

        blacklists_valid = Step(name='blacklists_valid',
                                transformer=blacklists,
                                input_data=['input'],
                                input_steps=[feature_by_type_split_valid],
                                adapter={'categorical_features': (
                                    [(feature_by_type_split_valid.name, 'categorical_features')])
                                },
                                cache_dirpath=config.env.cache_dirpath,
                                **kwargs)

        return blacklists, blacklists_valid

    else:
        feature_by_type_split = dispatchers
        blacklists = Step(name='blacklists',
                          transformer=fe.Blacklist(**config.blacklist),
                          input_data=['input'],
                          input_steps=[feature_by_type_split],
                          adapter={
                              'categorical_features': ([(feature_by_type_split.name, 'categorical_features')])
                          },
                          cache_dirpath=config.env.cache_dirpath,
                          **kwargs)

        return blacklists


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
                   config, train_mode, **kwargs):
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
                              cache_dirpath=config.env.cache_dirpath, **kwargs)

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
                                    cache_dirpath=config.env.cache_dirpath, **kwargs)

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
                              cache_dirpath=config.env.cache_dirpath, **kwargs)

        return feature_joiner


PIPELINES = {'baseline': {'train': partial(baseline, train_mode=True),
                          'inference': partial(baseline, train_mode=False)},
             'features_v1_lgbm': {'train': partial(features_v1_lgbm, train_mode=True),
                                  'inference': partial(features_v1_lgbm, train_mode=False)},
             'features_v2_lgbm': {'train': partial(features_v2_lgbm, train_mode=True),
                                  'inference': partial(features_v2_lgbm, train_mode=False)},
             'features_v2_xgboost': {'train': partial(features_v2_xgboost, train_mode=True),
                                     'inference': partial(features_v2_xgboost, train_mode=False)},
             'features_v2_log_reg': {'train': partial(features_v2_log_reg, train_mode=True),
                                     'inference': partial(features_v2_log_reg, train_mode=False)},
             }
