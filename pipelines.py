from functools import partial

from steps.base import Step, Dummy, to_list_inputs
from feature_extraction import FeatureDispatcher, FeatureJoiner, TargetEncoder, BinaryEncoder
from steps.misc import LightGBM


def baseline(config, train_mode=True):
    if train_mode:
        feature_dispatcher, feature_dispatcher_valid = _get_feature_dispatchers(config, train_mode)
        light_gbm = Step(name='light_gbm',
                         transformer=LightGBM(**config.light_gbm),
                         input_data=['input'],
                         input_steps=[feature_dispatcher, feature_dispatcher_valid],
                         adapter={'X': ([(feature_dispatcher.name, 'categorical_features')]),
                                  'y': ([('input', 'y')], to_numpy_label),
                                  'feature_names': ([(feature_dispatcher.name, 'categorical_feature_names')]),
                                  'categorical_features': ([(feature_dispatcher.name, 'categorical_feature_names')]),
                                  'X_valid': ([(feature_dispatcher_valid.name, 'categorical_features')]),
                                  'y_valid': ([('input', 'y_valid')], to_numpy_label),
                                  },
                         cache_dirpath=config.env.cache_dirpath)
    else:
        feature_dispatcher = _get_feature_dispatchers(config, train_mode)
        light_gbm = Step(name='light_gbm',
                         transformer=LightGBM(**config.light_gbm),
                         input_steps=[feature_dispatcher],
                         adapter={'X': ([(feature_dispatcher.name, 'categorical_features')]),
                                  },
                         cache_dirpath=config.env.cache_dirpath)

    output = Step(name='output',
                  transformer=Dummy(),
                  input_steps=[light_gbm],
                  adapter={'y_pred': ([(light_gbm.name, 'prediction')]),
                           },
                  cache_dirpath=config.env.cache_dirpath)
    return output


def categorical_encodings(config, train_mode=True):
    if train_mode:
        feature_dispatcher, feature_dispatcher_valid = _get_feature_dispatchers(config, train_mode)
        target_encoder, target_encoder_valid = _get_target_encoders([feature_dispatcher, feature_dispatcher_valid],
                                                                    config, train_mode)
        binary_encoder, binary_encoder_valid = _get_binary_encoders([feature_dispatcher, feature_dispatcher_valid],
                                                                    config, train_mode)

        feature_combiner, feature_combiner_valid = _join_features(numerical_features=[target_encoder, binary_encoder],
                                                                  numerical_features_valid=[target_encoder_valid,
                                                                                            binary_encoder_valid],
                                                                  categorical_features=[],
                                                                  categorical_features_valid=[],
                                                                  config=config, train_mode=train_mode)

        light_gbm = Step(name='light_gbm',
                         transformer=LightGBM(**config.light_gbm),
                         input_data=['input'],
                         input_steps=[feature_combiner, feature_combiner_valid],
                         adapter={'X': ([(feature_combiner.name, 'X')]),
                                  'y': ([('input', 'y')], to_numpy_label),
                                  'feature_names': ([(feature_combiner.name, 'feature_names')]),
                                  'categorical_features': ([(feature_combiner.name, 'categorical_features')]),
                                  'X_valid': ([(feature_combiner_valid.name, 'X')]),
                                  'y_valid': ([('input', 'y_valid')], to_numpy_label),
                                  },
                         cache_dirpath=config.env.cache_dirpath)
    else:
        feature_dispatcher = _get_feature_dispatchers(config, train_mode)
        target_encoder = _get_target_encoders(feature_dispatcher, config, train_mode)
        binary_encoder = _get_binary_encoders(feature_dispatcher, config, train_mode)

        feature_combiner = _join_features(numerical_features=[target_encoder, binary_encoder],
                                          numerical_features_valid=[],
                                          categorical_features=[],
                                          categorical_features_valid=[],
                                          config=config, train_mode=train_mode)

        light_gbm = Step(name='light_gbm',
                         transformer=LightGBM(**config.light_gbm),
                         input_steps=[feature_combiner],
                         adapter={'X': ([(feature_combiner.name, 'X')]),
                                  },
                         cache_dirpath=config.env.cache_dirpath)

    output = Step(name='output',
                  transformer=Dummy(),
                  input_steps=[light_gbm],
                  adapter={'y_pred': ([(light_gbm.name, 'prediction')]),
                           },
                  cache_dirpath=config.env.cache_dirpath)
    return output


def target_encoding(config, train_mode=True):
    if train_mode:
        feature_dispatcher, feature_dispatcher_valid = _get_feature_dispatchers(config, train_mode)
        target_encoder, target_encoder_valid = _get_target_encoders([feature_dispatcher, feature_dispatcher_valid],
                                                                    config, train_mode)

        light_gbm = Step(name='light_gbm',
                         transformer=LightGBM(**config.light_gbm),
                         input_data=['input'],
                         input_steps=[target_encoder, target_encoder_valid],
                         adapter={'X': ([(target_encoder.name, 'X')]),
                                  'y': ([('input', 'y')], to_numpy_label),
                                  'X_valid': ([(target_encoder_valid.name, 'X')]),
                                  'y_valid': ([('input', 'y_valid')], to_numpy_label),
                                  },
                         cache_dirpath=config.env.cache_dirpath)
    else:
        feature_dispatcher = _get_feature_dispatchers(config, train_mode)
        target_encoder = _get_target_encoders(feature_dispatcher, config, train_mode)

        light_gbm = Step(name='light_gbm',
                         transformer=LightGBM(**config.light_gbm),
                         input_steps=[target_encoder],
                         adapter={'X': ([(target_encoder.name, 'X')]),
                                  },
                         cache_dirpath=config.env.cache_dirpath)

    output = Step(name='output',
                  transformer=Dummy(),
                  input_steps=[light_gbm],
                  adapter={'y_pred': ([(light_gbm.name, 'prediction')]),
                           },
                  cache_dirpath=config.env.cache_dirpath)
    return output


def binary_encoding(config, train_mode=True):
    if train_mode:
        feature_dispatcher, feature_dispatcher_valid = _get_feature_dispatchers(config, train_mode)
        binary_encoder, binary_encoder_valid = _get_binary_encoders([feature_dispatcher, feature_dispatcher_valid],
                                                                    config, train_mode)

        light_gbm = Step(name='light_gbm',
                         transformer=LightGBM(**config.light_gbm),
                         input_data=['input'],
                         input_steps=[binary_encoder, binary_encoder_valid],
                         adapter={'X': ([(binary_encoder.name, 'X')]),
                                  'y': ([('input', 'y')], to_numpy_label),
                                  'X_valid': ([(binary_encoder_valid.name, 'X')]),
                                  'y_valid': ([('input', 'y_valid')], to_numpy_label),
                                  },
                         cache_dirpath=config.env.cache_dirpath)
    else:
        feature_dispatcher = _get_feature_dispatchers(config, train_mode)
        binary_encoder = _get_binary_encoders(feature_dispatcher, config, train_mode)

        light_gbm = Step(name='light_gbm',
                         transformer=LightGBM(**config.light_gbm),
                         input_steps=[binary_encoder],
                         adapter={'X': ([(binary_encoder.name, 'X')]),
                                  },
                         cache_dirpath=config.env.cache_dirpath)

    output = Step(name='output',
                  transformer=Dummy(),
                  input_steps=[light_gbm],
                  adapter={'y_pred': ([(light_gbm.name, 'prediction')]),
                           },
                  cache_dirpath=config.env.cache_dirpath)
    return output


def _get_feature_dispatchers(config, train_mode):
    if train_mode:
        feature_dispatcher = Step(name='feature_dispatcher',
                                  transformer=FeatureDispatcher(**config.feature_dispatcher),
                                  input_data=['input'],
                                  adapter={'X': ([('input', 'X')]),
                                           },
                                  cache_dirpath=config.env.cache_dirpath)

        feature_dispatcher_valid = Step(name='feature_dispatcher_valid',
                                        transformer=feature_dispatcher,
                                        input_data=['input'],
                                        adapter={'X': ([('input', 'X_valid')]),
                                                 },
                                        cache_dirpath=config.env.cache_dirpath)

        return feature_dispatcher, feature_dispatcher_valid

    else:
        feature_dispatcher = Step(name='feature_dispatcher',
                                  transformer=FeatureDispatcher(**config.feature_dispatcher),
                                  input_data=['input'],
                                  adapter={'X': ([('input', 'X')]),
                                           },
                                  cache_dirpath=config.env.cache_dirpath)

        return feature_dispatcher


def _get_target_encoders(dispatchers, config, train_mode, save_output=False):
    if train_mode:
        feature_dispatcher, feature_dispatcher_valid = dispatchers
        target_encoder = Step(name='target_encoder',
                              transformer=TargetEncoder(**config.target_encoder),
                              input_data=['input'],
                              input_steps=[feature_dispatcher],
                              adapter={'X': ([(feature_dispatcher.name, 'categorical_features')]),
                                       'y': ([('input', 'y')], to_numpy_label),
                                       },
                              cache_dirpath=config.env.cache_dirpath,
                              save_output=save_output)

        target_encoder_valid = Step(name='target_encoder_valid',
                                    transformer=target_encoder,
                                    input_data=['input'],
                                    input_steps=[feature_dispatcher_valid],
                                    adapter={'X': ([(feature_dispatcher_valid.name, 'categorical_features')]),
                                             'y': ([('input', 'y_valid')], to_numpy_label),
                                             },
                                    cache_dirpath=config.env.cache_dirpath,
                                    save_output=save_output)

        return target_encoder, target_encoder_valid

    else:
        feature_dispatcher = dispatchers

        target_encoder = Step(name='target_encoder',
                              transformer=TargetEncoder(**config.target_encoder),
                              input_data=['input'],
                              input_steps=[feature_dispatcher],
                              adapter={'X': ([(feature_dispatcher.name, 'categorical_features')]),
                                       },
                              cache_dirpath=config.env.cache_dirpath,
                              save_output=save_output)

        return target_encoder


def _get_binary_encoders(dispatchers, config, train_mode, save_output=False):
    if train_mode:
        feature_dispatcher, feature_dispatcher_valid = dispatchers
        binary_encoder = Step(name='binary_encoder',
                              transformer=BinaryEncoder(),
                              input_data=['input'],
                              input_steps=[feature_dispatcher],
                              adapter={'X': ([(feature_dispatcher.name, 'categorical_features')]),
                                       'y': ([('input', 'y')], to_numpy_label),
                                       },
                              cache_dirpath=config.env.cache_dirpath,
                              save_output=save_output)

        binary_encoder_valid = Step(name='binary_encoder_valid',
                                    transformer=binary_encoder,
                                    input_data=['input'],
                                    input_steps=[feature_dispatcher_valid],
                                    adapter={'X': ([(feature_dispatcher_valid.name, 'categorical_features')]),
                                             'y': ([('input', 'y_valid')], to_numpy_label),
                                             },
                                    cache_dirpath=config.env.cache_dirpath,
                                    save_output=save_output)

        return binary_encoder, binary_encoder_valid

    else:
        feature_dispatcher = dispatchers

        binary_encoder = Step(name='binary_encoder',
                              transformer=BinaryEncoder(),
                              input_data=['input'],
                              input_steps=[feature_dispatcher],
                              adapter={'X': ([(feature_dispatcher.name, 'categorical_features')]),
                                       },
                              cache_dirpath=config.env.cache_dirpath,
                              save_output=save_output)

        return binary_encoder


def _join_features(numerical_features, numerical_features_valid,
                   categorical_features, categorical_features_valid,
                   config, train_mode=False, save_output=False):
    if train_mode:
        feature_joiner = Step(name='feature_joiner',
                              transformer=FeatureJoiner(**config.feature_joiner),
                              input_steps=numerical_features + categorical_features,
                              adapter={
                                  'numerical_feature_list': (
                                  [(feature.name, 'X') for feature in numerical_features], to_list_inputs),
                                  'categorical_feature_list': (
                                      [(feature.name, 'X') for feature in categorical_features], to_list_inputs),
                              },
                              cache_dirpath=config.env.cache_dirpath,
                              save_output=save_output)

        feature_joiner_valid = Step(name='feature_joiner_valid',
                                    transformer=feature_joiner,
                                    input_steps=numerical_features_valid + categorical_features_valid,
                                    adapter={'numerical_feature_list': (
                                        [(feature.name, 'X') for feature in numerical_features_valid], to_list_inputs),
                                        'categorical_feature_list': (
                                            [(feature.name, 'X') for feature in categorical_features_valid],
                                            to_list_inputs),
                                    },
                                    cache_dirpath=config.env.cache_dirpath,
                                    save_output=save_output)

        return feature_joiner, feature_joiner_valid

    else:
        feature_joiner = Step(name='feature_joiner',
                              transformer=FeatureJoiner(**config.feature_joiner),
                              input_steps=numerical_features + categorical_features,
                              adapter={
                                  'numerical_feature_list': (
                                  [(feature.name, 'X') for feature in numerical_features], to_list_inputs),
                                  'categorical_feature_list': (
                                      [(feature.name, 'X') for feature in categorical_features], to_list_inputs),
                              },
                              cache_dirpath=config.env.cache_dirpath,
                              save_output=save_output)

        return feature_joiner


def to_numpy_label(inputs):
    return inputs[0].values.reshape(-1)


PIPELINES = {'baseline': {'train': partial(baseline, train_mode=True),
                          'inference': partial(baseline, train_mode=False)},
             'categorical_encodings': {'train': partial(categorical_encodings, train_mode=True),
                                       'inference': partial(categorical_encodings, train_mode=False)},
             'target_encoding': {'train': partial(target_encoding, train_mode=True),
                                 'inference': partial(target_encoding, train_mode=False)},
             'binary_encoding': {'train': partial(binary_encoding, train_mode=True),
                                 'inference': partial(binary_encoding, train_mode=False)},
             }
