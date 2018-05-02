import os

from attrdict import AttrDict
from deepsense import neptune

from utils import read_params, safe_eval

ctx = neptune.Context()
params = read_params(ctx)

FEATURE_COLUMNS = ['ip', 'app', 'device', 'os', 'channel', 'click_time']
TARGET_COLUMNS = ['is_attributed']
CV_COLUMNS = ['click_time']
ID_COLUMN = ['click_id']

DEV_TRAIN_DAYS = [8]
DEV_TRAIN_HOURS = [4, 5]
DEV_VALID_DAYS = [9]
DEV_VALID_HOURS = [4]
DEV_SAMPLE_TRAIN_SIZE = int(20e4)
DEV_SAMPLE_VALID_SIZE = int(10e4)
DEV_SAMPLE_TEST_SIZE = int(10e3)

COLUMN_TYPES = {'train': {'ip': 'uint32',
                          'app': 'uint16',
                          'device': 'uint16',
                          'os': 'uint16',
                          'channel': 'uint16',
                          'is_attributed': 'uint8',
                          },
                'inference': {'ip': 'uint32',
                              'app': 'uint16',
                              'device': 'uint16',
                              'os': 'uint16',
                              'channel': 'uint16',
                              'click_id': 'uint32'
                              }
                }

SOLUTION_CONFIG = AttrDict({
    'env': {'cache_dirpath': params.experiment_dir
            },
    'random_search': {'light_gbm': {'n_runs': safe_eval(params.lgbm_random_search_runs),
                                    'callbacks': {'neptune_monitor': {'name': 'light_gbm'
                                                                      },
                                                  'save_results': {'filepath': os.path.join(params.experiment_dir,
                                                                                            'random_search_light_gbm.pkl')
                                                                   }
                                                  }
                                    }
                      },
    'dataframe_by_type_splitter': {'numerical_columns': [],
                                   'categorical_columns': ['ip', 'app', 'device', 'os', 'channel'],
                                   'timestamp_columns': ['click_time'],
                                   },

    'time_delta': {'groupby_specs': [['app', 'os'], ['os', 'ip']],
                   'timestamp_column': 'click_time'
                   },

    'groupby_aggregation': {'groupby_aggregations': [
        {'groupby': ['app'], 'select': 'channel', 'agg': 'count'},
        {'groupby': ['ip','app','channel'], 'select': 'os', 'agg': 'mean'},
        {'groupby': ['app'], 'select': 'channel', 'agg': 'nunique'},
        {'groupby': ['ip', 'device', 'os'], 'select': 'app', 'agg': 'nunique'},
        {'groupby': ['ip','os','channel'], 'select': 'device', 'agg': 'var'}
    ]},

    'blacklist': {'blacklist': {'ip': [299172, 144604, 135992, 49386, 151908],
                                'app': [151, 56, 183, 93],
                                'device': [5, 182, 1728],
                                'os': [56, 65, 39, 79, 97],
                                'channel': [404, 420, 474]
                                }
                  },

    'confidence_rate': {'categories': [['app', 'os'], ['os', 'ip']],
                        'confidence_level': 100},

    'categorical_filter': {'categorical_columns': ['ip', 'app', 'device', 'os', 'channel'],
                           'min_frequencies': [20, 10, 10, 10, 10],
                           'impute_value': -1
                           },
    'target_encoder': {'n_splits': safe_eval(params.target_encoder__n_splits),
                       },
    'light_gbm': {'boosting_type': safe_eval(params.lgbm__boosting_type),
                  'objective': safe_eval(params.lgbm__objective),
                  'metric': safe_eval(params.lgbm__metric),
                  'learning_rate': safe_eval(params.lgbm__learning_rate),
                  'max_depth': safe_eval(params.lgbm__max_depth),
                  'subsample': safe_eval(params.lgbm__subsample),
                  'colsample_bytree': safe_eval(params.lgbm__colsample_bytree),
                  'min_child_weight': safe_eval(params.lgbm__min_child_weight),
                  'reg_lambda': safe_eval(params.lgbm__reg_lambda),
                  'reg_alpha': safe_eval(params.lgbm__reg_alpha),
                  'scale_pos_weight': safe_eval(params.lgbm__scale_pos_weight),
                  'subsample_freq': safe_eval(params.lgbm__subsample_freq),
                  'max_bin': safe_eval(params.lgbm__max_bin),
                  'min_child_samples': safe_eval(params.lgbm__min_child_samples),
                  'num_leaves': safe_eval(params.lgbm__num_leaves),
                  'nthread': safe_eval(params.num_workers),
                  'number_boosting_rounds': safe_eval(params.lgbm__number_boosting_rounds),
                  'early_stopping_rounds': safe_eval(params.lgbm__early_stopping_rounds),
                  'verbose': safe_eval(params.verbose)
                  },
})
