from itertools import product
import logging
import random
import sys
import os

from attrdict import AttrDict
from deepsense import neptune
import glob
import numpy as np
import pandas as pd
from tqdm import tqdm
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


def cut_data_in_time_chunks(data, timestamp_column, chunks_dir, logger=None):
    data[timestamp_column] = pd.to_datetime(data[timestamp_column], format='%Y-%m-%d %H:%M:%S')
    times = pd.DatetimeIndex(data[timestamp_column])
    grouped_train = data.groupby([times.day, times.hour])
    for (day, hour), train_chunk in grouped_train:
        chunk_filename = 'train_day{}_hour{}.csv'.format(day, hour)
        if logger is not None:
            logger.info('saving {}'.format(chunk_filename))
        else:
            print('saving {}'.format(chunk_filename))
        chunk_filepath = os.path.join(chunks_dir, chunk_filename)
        train_chunk.to_csv(chunk_filepath, index=None)


def read_csv_time_chunks(chunks_dir, days=[], hours=[], usecols=None, dtype=None, logger=None):
    filepaths = []
    for day, hour in product(days, hours):
        filepaths.extend(glob.glob('{}/train_day{}_hour{}.csv'.format(chunks_dir, day, hour)))
    data_chunks = []
    for filepath in tqdm(filepaths):
        data_chunk = pd.read_csv(filepath, usecols=usecols, dtype=dtype)
        if logger is not None:
            logger.info('read in chunk {} of shape {}'.format(filepath, data_chunk.shape))
        else:
            print('read in chunk {} of shape {}'.format(filepath, data_chunk.shape))
        data_chunks.append(data_chunk)
    data_chunks = pd.concat(data_chunks, axis=0)
    return data_chunks


def data_hash_channel_send(ctx, name, data):
    hash_channel = ctx.create_channel(name=name, channel_type=neptune.ChannelType.TEXT)
    data_hash = create_data_hash(data)
    hash_channel.send(y=data_hash)


def create_data_hash(data):
    if isinstance(data, pd.DataFrame) or isinstance(data, pd.Series):
        data_hash = make_pandas_hash(data)
    else:
        raise NotImplementedError('only pandas.DataFrame and pandas.Series are supported')
    return str(data_hash)


def make_pandas_hash(item):
    if isinstance(item, pd.DataFrame):
        index = tuple(item.index)
        columns = tuple(item.columns)
        values = tuple(tuple(x) for x in item.values)
        item = tuple([index, columns, values])
    elif isinstance(item, pd.Series):
        index = tuple(item.index)
        values = tuple(tuple(x) for x in item.values)
        item = tuple([index, values])

    try:
        return hash(item)
    except TypeError:
        try:
            # this might act funny if a thing is convertible to tuple but the tuple
            # is not a proper representation for the item (like for a frame :-()
            return hash(tuple(item))
        except TypeError as e:
            print("Unhashable type: %s, %s" % (item, [type(t) for t in tuple(item)]))
            raise e
