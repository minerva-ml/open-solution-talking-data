from attrdict import AttrDict
from deepsense import neptune

from utils import read_params

ctx = neptune.Context()
params = read_params(ctx)

FEATURE_COLUMNS = ['ip', 'app', 'device', 'os', 'channel', 'click_time']
TARGET_COLUMNS = ['is_attributed']
CV_COLUMNS = ['click_time']
ID_COLUMN = ['click_id']

DEV_TRAIN_DAYS = [8]
DEV_TRAIN_HOURS = [4]
DEV_VALID_DAYS = [9]
DEV_VALID_HOURS = [4]
DEV_SAMPLE_SIZE = int(10e4)

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
                              'click_id': 'uint32'}
                }

SOLUTION_CONFIG = AttrDict({
    'env': {'cache_dirpath': params.experiment_dir},

    'feature_dispatcher': {'numerical_columns': [],
                           'categorical_columns': ['app', 'device', 'os', 'channel'],
                           'timestamp_columns': ['click_time']
                           },

    'light_gbm': {'model_config': {'boosting_type': params.lgbm__boosting_type,
                                   'objective': params.lgbm__objective,
                                   'metric': params.lgbm__metric,
                                   'learning_rate': params.lgbm__learning_rate,
                                   'max_depth': params.lgbm__max_depth,
                                   'subsample': params.lgbm__subsample,
                                   'colsample_bytree': params.lgbm__colsample_bytree,
                                   'min_child_weight': params.lgbm__min_child_weight,
                                   'reg_lambda': params.lgbm__reg_lambda,
                                   'nthread': params.num_workers,
                                   'verbose': params.verbose},
                  'training_config': {'number_boosting_rounds': params.lgbm__number_boosting_rounds,
                                      'early_stopping_rounds': params.lgbm__early_stopping_rounds}
                  }
})
