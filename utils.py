import logging
import random
import sys
import os
import math
from collections import deque
from io import StringIO

import numpy as np
from sklearn.externals import joblib
import pandas as pd
import yaml
from attrdict import AttrDict


def read_yaml(filepath):
    with open(filepath) as f:
        config = yaml.load(f)
    return AttrDict(config)


def init_logger():
    logger = logging.getLogger('talking-data')
    logger.setLevel(logging.INFO)
    message_format = logging.Formatter(fmt='%(asctime)s %(name)s >>> %(message)s',
                                       datefmt='%Y-%m-%d %H-%M-%S')

    # console handler for validation info
    ch_va = logging.StreamHandler(sys.stdout)
    ch_va.setLevel(logging.INFO)

    ch_va.setFormatter(fmt=message_format)

    # add the handlers to the logger
    logger.addHandler(ch_va)

    return logger


def get_logger():
    return logging.getLogger('talking-data')


def create_submission(meta, predictions):
    submission = pd.DataFrame({'click_id': meta['click_id'].tolist(),
                               'is_attributed': predictions
                               })
    return submission


def read_params(ctx):
    if ctx.params.__class__.__name__ == 'OfflineContextParams':
        neptune_config = read_yaml('neptune.yaml')
        params = neptune_config.parameters
    else:
        params = ctx.params
    return params


def squeeze_inputs(inputs):
    return np.squeeze(inputs[0], axis=1)


def sigmoid(x):
    return 1. / (1 + np.exp(-x))


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def train_valid_split_on_timestamp(meta, validation_size, timestamp_column, sort=True, shuffle=True, random_state=1234):
    n_rows = len(meta)
    train_size = n_rows - math.floor(n_rows * validation_size)
    if sort:
        meta.sort_values(timestamp_column, inplace=True)
    meta_train_split = meta.iloc[:train_size]
    meta_valid_split = meta.iloc[train_size:]

    if shuffle:
        meta_train_split = meta_train_split.sample(frac=1, random_state=random_state)
        meta_valid_split = meta_valid_split.sample(frac=1, random_state=random_state)

    print('Target distribution in train: {}'.format(meta_train_split['is_attributed'].mean()))
    print('Target distribution in valid: {}'.format(meta_valid_split['is_attributed'].mean()))

    return meta_train_split, meta_valid_split


def read_csv_last_n_rows(filepath, nrows):
    columns = pd.read_csv(filepath, nrows=1).columns
    with open(filepath, 'r') as f:
        q = deque(f, nrows)
    df = pd.read_csv(StringIO(''.join(q)), header=None)
    df.columns = columns
    return df


def save_worst_predictions(experiment_dir, y_true, y_pred, raw_data, worst_n, eps=1e-15):
    y_pred = np.clip(y_pred, eps, 1 - eps)
    scores = y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)
    scores_df = pd.DataFrame({'score': scores})

    worst_n_scores = scores_df.sort_values('score').iloc[:worst_n]
    worst_n_indeces = list(worst_n_scores.index)

    raw_data.reset_index(drop=True, inplace=True)
    raw_data_worst = raw_data.iloc[worst_n_indeces]

    y_pred_worst = y_pred[worst_n_indeces]
    y_true_worst = y_true[worst_n_indeces]

    worst = {'scores': worst_n_scores.values,
             'data': raw_data_worst,
             'y_pred': y_pred_worst,
             'y_true': y_true_worst
             }

    filepath = os.path.join(experiment_dir, 'worst_predictions.pkl')
    joblib.dump(worst, filepath)
