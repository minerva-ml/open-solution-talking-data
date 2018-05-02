import glob
import hashlib
import logging
import os
import random
import sys
from itertools import product

import numpy as np
import pandas as pd
import yaml
from attrdict import AttrDict
from deepsense import neptune
from tqdm import tqdm


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


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


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


def cut_data_in_time_chunks(data, timestamp_column, chunks_dir, prefix='chunk', logger=None):
    data[timestamp_column] = pd.to_datetime(data[timestamp_column], format='%Y-%m-%d %H:%M:%S')
    times = pd.DatetimeIndex(data[timestamp_column])
    grouped_train = data.groupby([times.day, times.hour])
    for (day, hour), train_chunk in grouped_train:
        chunk_filename = '{}_day{}_hour{}.csv'.format(prefix, day, hour)
        if logger is not None:
            logger.info('saving {}'.format(chunk_filename))
        else:
            print('saving {}'.format(chunk_filename))
        chunk_filepath = os.path.join(chunks_dir, chunk_filename)
        train_chunk.to_csv(chunk_filepath, index=None)


def read_csv_time_chunks(chunks_dir, prefix='chunk', days_hours={}, usecols=None, dtype=None, logger=None):
    filepaths = []
    for day, hours in days_hours.items():
        for hour in hours:
            filepaths.extend(glob.glob('{}/{}_day{}_hour{}.csv'.format(chunks_dir, prefix, day, hour)))
    data_chunks = []
    for filepath in tqdm(filepaths):
        data_chunk = pd.read_csv(filepath, usecols=usecols, dtype=dtype)
        if logger is not None:
            logger.info('read in chunk {} of shape {}'.format(filepath, data_chunk.shape))
        else:
            print('read in chunk {} of shape {}'.format(filepath, data_chunk.shape))
        data_chunks.append(data_chunk)
    data_chunks = pd.concat(data_chunks, axis=0).reset_index(drop=True)
    data_chunks['click_time'] = pd.to_datetime(data_chunks['click_time'], format='%Y-%m-%d %H:%M:%S')

    if logger is not None:
        logger.info('combined dataset shape: {}'.format(data_chunks.shape))
    else:
        print('combined dataset shape: {}'.format(data_chunks.shape))
    return data_chunks


def data_hash_channel_send(ctx, name, data):
    hash_channel = ctx.create_channel(name=name, channel_type=neptune.ChannelType.TEXT)
    data_hash = create_data_hash(data)
    hash_channel.send(y=data_hash)


def create_data_hash(data):
    if isinstance(data, pd.DataFrame):
        data_hash = hashlib.sha256(data.to_json().encode()).hexdigest()
    else:
        raise NotImplementedError('only pandas.DataFrame and pandas.Series are supported')
    return str(data_hash)


def safe_eval(obj):
    try:
        return eval(obj)
    except Exception:
        return obj


def get_submission_hours_index(meta, timestamp_column, submission_hours):
    times = pd.DatetimeIndex(meta[timestamp_column])
    filtered_indeces = []
    for hour in submission_hours:
        chunk = np.where(times.hour == hour)
        filtered_indeces.extend(chunk[0])
    return filtered_indeces
