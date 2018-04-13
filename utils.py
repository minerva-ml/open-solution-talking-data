import logging
import random
import sys
import math

import numpy as np
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


def train_valid_split_on_timestamp(meta, validation_size, timestamp_column, shuffle=True, random_state=1234):
    n_rows = len(meta)
    train_size = n_rows - math.floor(n_rows * validation_size)

    meta = meta.sort_values(timestamp_column)

    meta_train_split = meta.iloc[:train_size]
    meta_valid_split = meta.iloc[train_size:]

    if shuffle:
        meta_train_split = meta_train_split.sample(frac=1, random_state=random_state)
        meta_valid_split = meta_valid_split.sample(frac=1, random_state=random_state)

    return meta_train_split, meta_valid_split
