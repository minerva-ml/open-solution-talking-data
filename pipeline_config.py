from attrdict import AttrDict
from deepsense import neptune

from utils import read_params

ctx = neptune.Context()
params = read_params(ctx)

FEATURE_COLUMNS = ['ip', 'app', 'device', 'os', 'channel', 'click_time']
TARGET_COLUMNS = ['is_attributed']
CV_COLUMNS = ['click_time']

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
