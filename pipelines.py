from functools import partial

from steps.base import Step, Dummy
from feature_extraction import FeatureDispatcher
from steps.misc import LightGBM


def baseline(config, train_mode=True):
    feature_dispatcher = Step(name='feature_dispatcher',
                              transformer=FeatureDispatcher(**config.feature_dispatcher),
                              input_data=['input'],
                              adapter={'X': ([('input', 'X')]),
                                       },
                              cache_dirpath=config.env.cache_dirpath)

    if train_mode:
        feature_dispatcher_valid = Step(name='feature_dispatcher_valid',
                                        transformer=feature_dispatcher,
                                        input_data=['input'],
                                        adapter={'X': ([('input', 'X_valid')]),
                                                 },
                                        cache_dirpath=config.env.cache_dirpath)

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


def to_numpy_label(inputs):
    return inputs[0].values.reshape(-1)


PIPELINES = {'baseline': {'train': partial(baseline, train_mode=True),
                             'inference': partial(baseline, train_mode=False)},
             }
