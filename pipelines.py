from functools import partial
from sklearn.metrics import roc_auc_score

from steps.adapters import to_numpy_label_inputs, identity_inputs
from steps.base import Step, Dummy
from steps.misc import LightGBM
import feature_extraction as fe
from hyperparameter_tuning import RandomSearchOptimizer, NeptuneMonitor, SaveResults


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
        features, features_valid = feature_extraction_v1(config, train_mode)
        light_gbm = classifier_lgbm((features, features_valid), config, train_mode)
    else:
        features = feature_extraction_v1(config, train_mode)
        light_gbm = classifier_lgbm(features, train_mode)

    output = Step(name='output',
                  transformer=Dummy(),
                  input_steps=[light_gbm],
                  adapter={'y_pred': ([(light_gbm.name, 'prediction')]),
                           },
                  cache_dirpath=config.env.cache_dirpath)
    return output


def feature_extraction_v0(config, train_mode):
    if train_mode:
        feature_by_type_split, feature_by_type_split_valid = _get_feature_by_type_splits(config, train_mode)
        categorical_features = Step(name='categorical_features',
                                    transformer=Dummy(),
                                    input_steps=[feature_by_type_split],
                                    adapter={'X': ([(feature_by_type_split.name, 'categorical_features')]),
                                             },
                                    cache_dirpath=config.env.cache_dirpath)
        categorical_features_valid = Step(name='categorical_features_valid',
                                          transformer=Dummy(),
                                          input_steps=[feature_by_type_split_valid],
                                          adapter={'X': ([(feature_by_type_split_valid.name, 'categorical_features')]),
                                                   },
                                          cache_dirpath=config.env.cache_dirpath)
        feature_combiner = _join_features(numerical_features=[],
                                          numerical_features_valid=[],
                                          categorical_features=[categorical_features],
                                          categorical_features_valid=[categorical_features_valid],
                                          config=config, train_mode=train_mode)
        return feature_combiner
    else:
        feature_by_type_split = _get_feature_by_type_splits(config, train_mode)
        categorical_features = Step(name='categorical_features',
                                    transformer=Dummy(),
                                    input_steps=[feature_by_type_split],
                                    adapter={'X': ([(feature_by_type_split.name, 'categorical_features')]),
                                             },
                                    cache_dirpath=config.env.cache_dirpath)
        feature_combiner = _join_features(numerical_features=[],
                                          numerical_features_valid=[],
                                          categorical_features=[categorical_features],
                                          categorical_features_valid=[],
                                          config=config, train_mode=train_mode)
        return feature_combiner


def feature_extraction_v1(config, train_mode):
    if train_mode:
        feature_by_type_split, feature_by_type_split_valid = _get_feature_by_type_splits(config, train_mode)
        filtered_categorical, filtered_categorical_valid = _get_categorical_frequency_filters(
            (feature_by_type_split, feature_by_type_split_valid),
            config, train_mode)

        target_encoder, target_encoder_valid = _get_target_encoders((filtered_categorical, filtered_categorical_valid),
                                                                    config, train_mode)
        binary_encoder, binary_encoder_valid = _get_binary_encoders((filtered_categorical, filtered_categorical_valid),
                                                                    config, train_mode)

        feature_combiner, feature_combiner_valid = _join_features(numerical_features=[target_encoder, binary_encoder],
                                                                  numerical_features_valid=[target_encoder_valid,
                                                                                            binary_encoder_valid],
                                                                  categorical_features=[],
                                                                  categorical_features_valid=[],
                                                                  config=config, train_mode=train_mode)
        return feature_combiner, feature_combiner_valid
    else:
        feature_by_type_split = _get_feature_by_type_splits(config, train_mode)

        filtered_categorical, filtered_categorical_valid = _get_categorical_frequency_filters(
            feature_by_type_split,
            config, train_mode)
        target_encoder = _get_target_encoders(filtered_categorical, config, train_mode)
        binary_encoder = _get_binary_encoders(filtered_categorical, config, train_mode)

        feature_combiner = _join_features(numerical_features=[target_encoder, binary_encoder],
                                          numerical_features_valid=[],
                                          categorical_features=[],
                                          categorical_features_valid=[],
                                          config=config, train_mode=train_mode)
    return feature_combiner


def classifier_lgbm(features, config, train_mode):
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
                         adapter={'X': ([(features_train.name, 'X')]),
                                  'y': ([('input', 'y')], to_numpy_label_inputs),
                                  'feature_names': ([(features_train.name, 'feature_names')]),
                                  'categorical_features': ([(features_train.name, 'categorical_features')]),
                                  'X_valid': ([(features_valid.name, 'X')]),
                                  'y_valid': ([('input', 'y_valid')], to_numpy_label_inputs),
                                  },
                         cache_dirpath=config.env.cache_dirpath)
    else:
        light_gbm = Step(name='light_gbm',
                         transformer=LightGBM(**config.light_gbm),
                         input_steps=[features],
                         adapter={'X': ([(features.name, 'X')]),
                                  },
                         cache_dirpath=config.env.cache_dirpath)
    return light_gbm


def _get_feature_by_type_splits(config, train_mode):
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


def _get_categorical_frequency_filters(dispatchers, config, train_mode, save_output=False):
    if train_mode:
        feature_by_type_split, feature_by_type_split_valid = dispatchers
        categorical_filter = Step(name='categorical_filter',
                                  transformer=fe.CategoricalFilter(**config.categorical_filter),
                                  input_steps=[feature_by_type_split],
                                  adapter={
                                      'categorical_features': ([(feature_by_type_split.name, 'categorical_features')]),
                                  },
                                  cache_dirpath=config.env.cache_dirpath,
                                  save_output=save_output)

        categorical_filter_valid = Step(name='categorical_filter_valid',
                                        transformer=categorical_filter,
                                        input_steps=[feature_by_type_split_valid],
                                        adapter={'categorical_features': (
                                            [(feature_by_type_split_valid.name, 'categorical_features')]),
                                        },
                                        cache_dirpath=config.env.cache_dirpath,
                                        save_output=save_output)

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
                                  save_output=save_output)

        return categorical_filter


def _get_target_encoders(dispatchers, config, train_mode, save_output=False):
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
                              save_output=save_output)

        target_encoder_valid = Step(name='target_encoder_valid',
                                    transformer=target_encoder,
                                    input_data=['input'],
                                    input_steps=[feature_by_type_split_valid],
                                    adapter={'X': ([(feature_by_type_split_valid.name, 'categorical_features')]),
                                             'y': ([('input', 'y_valid')], to_numpy_label_inputs),
                                             },
                                    cache_dirpath=config.env.cache_dirpath,
                                    save_output=save_output)

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
                              save_output=save_output)

        return target_encoder


def _get_binary_encoders(dispatchers, config, train_mode, save_output=False):
    if train_mode:
        feature_by_type_split, feature_by_type_split_valid = dispatchers
        binary_encoder = Step(name='binary_encoder',
                              transformer=fe.BinaryEncoder(),
                              input_data=['input'],
                              input_steps=[feature_by_type_split],
                              adapter={'X': ([(feature_by_type_split.name, 'categorical_features')]),
                                       'y': ([('input', 'y')], to_numpy_label_inputs),
                                       },
                              cache_dirpath=config.env.cache_dirpath,
                              save_output=save_output)

        binary_encoder_valid = Step(name='binary_encoder_valid',
                                    transformer=binary_encoder,
                                    input_data=['input'],
                                    input_steps=[feature_by_type_split_valid],
                                    adapter={'X': ([(feature_by_type_split_valid.name, 'categorical_features')]),
                                             'y': ([('input', 'y_valid')], to_numpy_label_inputs),
                                             },
                                    cache_dirpath=config.env.cache_dirpath,
                                    save_output=save_output)

        return binary_encoder, binary_encoder_valid

    else:
        feature_by_type_split = dispatchers

        binary_encoder = Step(name='binary_encoder',
                              transformer=fe.BinaryEncoder(),
                              input_data=['input'],
                              input_steps=[feature_by_type_split],
                              adapter={'X': ([(feature_by_type_split.name, 'categorical_features')]),
                                       },
                              cache_dirpath=config.env.cache_dirpath,
                              save_output=save_output)

        return binary_encoder


def _join_features(numerical_features, numerical_features_valid,
                   categorical_features, categorical_features_valid,
                   config, train_mode=False, save_output=False):
    if train_mode:
        feature_joiner = Step(name='feature_joiner',
                              transformer=fe.FeatureJoiner(),
                              input_steps=numerical_features + categorical_features,
                              adapter={
                                  'numerical_feature_list': (
                                      [(feature.name, 'X') for feature in numerical_features], identity_inputs),
                                  'categorical_feature_list': (
                                      [(feature.name, 'X') for feature in categorical_features], identity_inputs),
                              },
                              cache_dirpath=config.env.cache_dirpath,
                              save_output=save_output)

        feature_joiner_valid = Step(name='feature_joiner_valid',
                                    transformer=feature_joiner,
                                    input_steps=numerical_features_valid + categorical_features_valid,
                                    adapter={'numerical_feature_list': (
                                        [(feature.name, 'X') for feature in numerical_features_valid], identity_inputs),
                                        'categorical_feature_list': (
                                            [(feature.name, 'X') for feature in categorical_features_valid],
                                            identity_inputs),
                                    },
                                    cache_dirpath=config.env.cache_dirpath,
                                    save_output=save_output)

        return feature_joiner, feature_joiner_valid

    else:
        feature_joiner = Step(name='feature_joiner',
                              transformer=fe.FeatureJoiner(),
                              input_steps=numerical_features + categorical_features,
                              adapter={
                                  'numerical_feature_list': (
                                      [(feature.name, 'X') for feature in numerical_features], identity_inputs),
                                  'categorical_feature_list': (
                                      [(feature.name, 'X') for feature in categorical_features], identity_inputs),
                              },
                              cache_dirpath=config.env.cache_dirpath,
                              save_output=save_output)

        return feature_joiner


PIPELINES = {'baseline': {'train': partial(baseline, train_mode=True),
                          'inference': partial(baseline, train_mode=False)},
             'solution_1': {'train': partial(solution_1, train_mode=True),
                            'inference': partial(solution_1, train_mode=False)},
             }
