import os

from attrdict import AttrDict
from deepsense import neptune

from utils import read_params, safe_eval

ctx = neptune.Context()
params = read_params(ctx)

FEATURE_COLUMNS = ['ip', 'app', 'device', 'os', 'channel', 'click_time']
TARGET_COLUMNS = ['is_attributed']
CV_COLUMNS = ['click_time']
TIMESTAMP_COLUMN = 'click_time'
ID_COLUMN = ['click_id']

DEV_TRAIN_DAYS_HOURS = {8: [4, 5]
                        }
DEV_VALID_DAYS_HOURS = {9: [4]
                        }
DEV_TEST_DAYS_HOURS = {10: [4]
                       }
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
                                    },
                      'xgboost': {'n_runs': safe_eval(params.xgboost_random_search_runs),
                                  'callbacks': {'neptune_monitor': {'name': 'xgboost'
                                                                    },
                                                'save_results': {'filepath': os.path.join(params.experiment_dir,
                                                                                          'random_search_xgboost.pkl')
                                                                 }
                                                }
                                  },
                      'log_reg': {'n_runs': safe_eval(params.log_reg_random_search_runs),
                                  'callbacks': {'neptune_monitor': {'name': 'log_reg'
                                                                    },
                                                'save_results': {'filepath': os.path.join(params.experiment_dir,
                                                                                          'random_search_log_regl.pkl')
                                                                 }
                                                }
                                  }
                      },
    'dataframe_by_type_splitter': {'numerical_columns': [],
                                   'categorical_columns': ['ip', 'app', 'device', 'os', 'channel'],
                                   'timestamp_columns': ['click_time'],
                                   },

    'time_delta': {'groupby_specs': [['ip', 'app', 'device', 'os']],
                   'timestamp_column': 'click_time'
                   },

    'groupby_aggregation': {'groupby_aggregations': [
        {'groupby': ['ip'], 'select': 'app', 'agg': 'count'},
        {'groupby': ['ip', 'app'], 'select': 'device', 'agg': 'count'},
        {'groupby': ['ip'], 'select': 'app', 'agg': 'nunique'},
        {'groupby': ['ip'], 'select': 'channel', 'agg': 'nunique'},
        {'groupby': ['ip', 'device', 'os'], 'select': 'app', 'agg': 'nunique'}
    ]},

    'blacklist': {'blacklist': {'ip': [299172, 144604, 135992, 49386, 151908],
                                'app': [151, 56, 183, 93],
                                'device': [5, 182, 1728],
                                'os': [56, 65, 39, 79, 97],
                                'channel': [404, 420, 474]
                                }
                  },

    'confidence_rate': {'categories': [['ip'], ['app'], ['ip', 'os'], ['app', 'os'], ['app', 'channel'],
                                       ['os', 'channel'], ['device', 'channel'], ['app', 'device']],
                        'confidence_level': 10000},

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

    'xgboost': {'temp_dir': os.path.join(params.experiment_dir, 'tmp'),
                'booster': safe_eval(params.xgboost__booster),
                'objective': safe_eval(params.xgboost__objective),
                'eval_metric': safe_eval(params.xgboost__eval_metric),
                'learning_rate': safe_eval(params.xgboost__learning_rate),
                'max_depth': safe_eval(params.xgboost__max_depth),
                'gamma': safe_eval(params.xgboost__gamma),
                'subsample': safe_eval(params.xgboost__subsample),
                'colsample_bytree': safe_eval(params.xgboost__colsample_bytree),
                'reg_lambda': safe_eval(params.xgboost__reg_lambda),
                'reg_alpha': safe_eval(params.xgboost__reg_alpha),
                'scale_pos_weight': safe_eval(params.xgboost__scale_pos_weight),
                'number_boosting_rounds': safe_eval(params.xgboost__number_boosting_rounds),
                'early_stopping_rounds': safe_eval(params.xgboost__early_stopping_rounds),
                'maximize': safe_eval(params.xgboost__maximize),
                'n_jobs': safe_eval(params.num_workers),
                'verbose': safe_eval(params.verbose)
                },

    'log_reg': {'solver': 'sag',
                'penalty': safe_eval(params.log_reg__penalty),
                'C': safe_eval(params.log_reg__C),
                'class_weight': safe_eval(params.log_reg__class_weight),
                'n_jobs': safe_eval(params.num_workers),
                'verbose': safe_eval(params.verbose)
                },
})
