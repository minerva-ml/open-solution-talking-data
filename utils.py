import logging
import random
import sys
import os
import math

from attrdict import AttrDict
import numpy as np
import pandas as pd
import yaml


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


def train_valid_split_on_timestamp(meta, validation_size, timestamp_column, sort=True):
    n_rows = len(meta)
    train_size = n_rows - math.floor(n_rows * validation_size)
    if sort:
        meta.sort_values(timestamp_column, inplace=True)
    meta_train_split = meta.iloc[:train_size]
    meta_valid_split = meta.iloc[train_size:]

    return meta_train_split, meta_valid_split


def log_loss_row(y_true, y_pred, eps=1e-15):
    y_pred = np.clip(y_pred, eps, 1 - eps)
    scores = y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)
    return scores


def save_evaluation_predictions(experiment_dir, y_true, y_pred, raw_data):
    raw_data['y_pred'] = y_pred
    raw_data['score'] = log_loss_row(y_true, y_pred)

    raw_data.sort_values('score', ascending=False, inplace=True)

    filepath = os.path.join(experiment_dir, 'evaluation_predictions.csv')
    raw_data.to_csv(filepath, index=None)

