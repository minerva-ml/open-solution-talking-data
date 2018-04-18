import os
import shutil

import click
import pandas as pd
from deepsense import neptune
from sklearn.metrics import roc_auc_score

import pipeline_config as cfg
from pipelines import PIPELINES
from utils import init_logger, read_params, create_submission, set_seed, \
    save_evaluation_predictions, read_csv_time_chunks, cut_data_in_time_chunks

set_seed(1234)
logger = init_logger()
ctx = neptune.Context()
params = read_params(ctx)


@click.group()
def action():
    pass


@action.command()
def prepare_data():
    train = pd.read_csv(params.raw_train_filepath)
    cut_data_in_time_chunks(train,
                            timestamp_column='click_time',
                            chunks_dir=params.train_chunks_dir,
                            logger=logger)


@action.command()
@click.option('-p', '--pipeline_name', help='pipeline to be trained', required=True)
@click.option('-d', '--dev_mode', help='if true only a small sample of data will be used', is_flag=True, required=False)
def train(pipeline_name, dev_mode):
    _train(pipeline_name, dev_mode)


def _train(pipeline_name, dev_mode):
    if bool(params.overwrite) and os.path.isdir(params.experiment_dir):
        shutil.rmtree(params.experiment_dir)

    logger.info('reading data in')
    if dev_mode:
        TRAIN_DAYS, TRAIN_HOURS = cfg.DEV_TRAIN_DAYS, cfg.DEV_TRAIN_HOURS
        VALID_DAYS, VALID_HOURS = cfg.DEV_VALID_DAYS, cfg.DEV_VALID_HOURS
    else:
        TRAIN_DAYS, TRAIN_HOURS = eval(params.train_days), eval(params.train_hours)
        VALID_DAYS, VALID_HOURS = eval(params.valid_days), eval(params.valid_hours)

    meta_train_split = read_csv_time_chunks(params.train_chunks_dir,
                                            days=TRAIN_DAYS,
                                            hours=TRAIN_HOURS,
                                            usecols=cfg.FEATURE_COLUMNS + cfg.TARGET_COLUMNS,
                                            dtype=cfg.COLUMN_TYPES['train'],
                                            logger=logger)
    meta_valid_split = read_csv_time_chunks(params.train_chunks_dir,
                                            days=VALID_DAYS,
                                            hours=VALID_HOURS,
                                            usecols=cfg.FEATURE_COLUMNS + cfg.TARGET_COLUMNS,
                                            dtype=cfg.COLUMN_TYPES['train'],
                                            logger=logger)

    if dev_mode:
        meta_train_split = meta_train_split.sample(cfg.DEV_SAMPLE_SIZE, replace=False)
        meta_valid_split = meta_valid_split.sample(cfg.DEV_SAMPLE_SIZE, replace=False)

    logger.info('Target distribution in train: {}'.format(meta_train_split['is_attributed'].mean()))
    logger.info('Target distribution in valid: {}'.format(meta_valid_split['is_attributed'].mean()))

    logger.info('shuffling data')
    meta_train_split = meta_train_split.sample(frac=1)
    meta_valid_split = meta_valid_split.sample(frac=1)

    data = {'input': {'X': meta_train_split[cfg.FEATURE_COLUMNS],
                      'y': meta_train_split[cfg.TARGET_COLUMNS],
                      'X_valid': meta_valid_split[cfg.FEATURE_COLUMNS],
                      'y_valid': meta_valid_split[cfg.TARGET_COLUMNS],
                      },
            }

    pipeline = PIPELINES[pipeline_name]['train'](cfg.SOLUTION_CONFIG)
    pipeline.clean_cache()
    pipeline.fit_transform(data)
    pipeline.clean_cache()


@action.command()
@click.option('-p', '--pipeline_name', help='pipeline to be trained', required=True)
@click.option('-d', '--dev_mode', help='if true only a small sample of data will be used', is_flag=True, required=False)
def evaluate(pipeline_name, dev_mode):
    _evaluate(pipeline_name, dev_mode)


def _evaluate(pipeline_name, dev_mode):
    logger.info('reading data in')
    if dev_mode:
        VALID_DAYS, VALID_HOURS = cfg.DEV_VALID_DAYS, cfg.DEV_VALID_HOURS
    else:
        VALID_DAYS, VALID_HOURS = eval(params.valid_days), eval(params.valid_hours)

    meta_valid_split = read_csv_time_chunks(params.train_chunks_dir,
                                            days=VALID_DAYS,
                                            hours=VALID_HOURS,
                                            usecols=cfg.FEATURE_COLUMNS + cfg.TARGET_COLUMNS,
                                            dtype=cfg.COLUMN_TYPES['train'],
                                            logger=logger)

    if dev_mode:
        meta_valid_split = meta_valid_split.sample(cfg.DEV_SAMPLE_SIZE, replace=False)

    logger.info('Target distribution in valid: {}'.format(meta_valid_split['is_attributed'].mean()))

    data = {'input': {'X': meta_valid_split[cfg.FEATURE_COLUMNS],
                      'y': None,
                      },
            }
    y_true = meta_valid_split[cfg.TARGET_COLUMNS].values.reshape(-1)

    pipeline = PIPELINES[pipeline_name]['inference'](cfg.SOLUTION_CONFIG)
    pipeline.clean_cache()
    output = pipeline.transform(data)
    pipeline.clean_cache()
    y_pred = output['y_pred']

    logger.info('Calculating ROC_AUC Scores')
    score = roc_auc_score(y_true, y_pred)
    logger.info('ROC_AUC score on validation is {}'.format(score))
    ctx.channel_send('ROC_AUC', 0, score)

    logger.info('Saving evaluation predictions')
    save_evaluation_predictions(params.experiment_dir, y_true, y_pred, meta_valid_split)


@action.command()
@click.option('-p', '--pipeline_name', help='pipeline to be trained', required=True)
@click.option('-d', '--dev_mode', help='if true only a small sample of data will be used', is_flag=True, required=False)
@click.option('-c', '--chunk_size', help='size of the chunks to run prediction on', type=int, default=None,
              required=False)
def predict(pipeline_name, dev_mode, chunk_size):
    if chunk_size is not None:
        _predict_in_chunks(pipeline_name, dev_mode, chunk_size)
    else:
        _predict(pipeline_name, dev_mode)


def _predict(pipeline_name, dev_mode):
    logger.info('reading data in')
    if dev_mode:
        meta_test = pd.read_csv(params.test_filepath,
                                usecols=cfg.FEATURE_COLUMNS + cfg.ID_COLUMN,
                                dtype=cfg.COLUMN_TYPES['inference'],
                                nrows=cfg.DEV_SAMPLE_SIZE)
    else:
        meta_test = pd.read_csv(params.test_filepath,
                                usecols=cfg.FEATURE_COLUMNS + cfg.ID_COLUMN,
                                dtype=cfg.COLUMN_TYPES['inference'])

    data = {'input': {'X': meta_test[cfg.FEATURE_COLUMNS],
                      'y': None,
                      },
            }

    pipeline = PIPELINES[pipeline_name]['inference'](cfg.SOLUTION_CONFIG)
    pipeline.clean_cache()
    output = pipeline.transform(data)
    pipeline.clean_cache()
    y_pred = output['y_pred']

    logger.info('creating submission')
    submission = create_submission(meta_test, y_pred)

    submission_filepath = os.path.join(params.experiment_dir, 'submission.csv')
    submission.to_csv(submission_filepath, index=None, encoding='utf-8')
    logger.info('submission saved to {}'.format(submission_filepath))
    logger.info('submission head \n\n{}'.format(submission.head()))


def _predict_in_chunks(pipeline_name, dev_mode, chunk_size):
    submission_chunks = []
    for meta_test_chunk in pd.read_csv(params.test_filepath, chunksize=chunk_size):
        data = {'input': {'meta': meta_test_chunk,
                          },
                }

        pipeline = PIPELINES[pipeline_name]['inference'](cfg.SOLUTION_CONFIG)
        pipeline.clean_cache()
        output = pipeline.transform(data)
        pipeline.clean_cache()
        y_pred = output['y_pred']

        submission_chunk = create_submission(meta_test_chunk, y_pred)
        submission_chunks.append(submission_chunk)

        if dev_mode:
            break

    submission = pd.concat(submission_chunks, axis=0)

    submission_filepath = os.path.join(params.experiment_dir, 'submission.csv')
    submission.to_csv(submission_filepath, index=None, encoding='utf-8')
    logger.info('submission saved to {}'.format(submission_filepath))
    logger.info('submission head \n\n{}'.format(submission.head()))


@action.command()
@click.option('-p', '--pipeline_name', help='pipeline to be trained', required=True)
@click.option('-d', '--dev_mode', help='if true only a small sample of data will be used', is_flag=True, required=False)
@click.option('-c', '--chunk_size', help='size of the chunks to run prediction on', type=int, default=None,
              required=False)
def train_evaluate_predict(pipeline_name, dev_mode, chunk_size):
    logger.info('training')
    _train(pipeline_name, dev_mode)
    logger.info('evaluate')
    _evaluate(pipeline_name, dev_mode)
    logger.info('predicting')
    if chunk_size is not None:
        _predict_in_chunks(pipeline_name, dev_mode, chunk_size)
    else:
        _predict(pipeline_name, dev_mode)


@action.command()
@click.option('-p', '--pipeline_name', help='pipeline to be trained', required=True)
@click.option('-d', '--dev_mode', help='if true only a small sample of data will be used', is_flag=True, required=False)
@click.option('-c', '--chunk_size', help='size of the chunks to run prediction on', type=int, default=None,
              required=False)
def evaluate_predict(pipeline_name, dev_mode, chunk_size):
    logger.info('evaluate')
    _evaluate(pipeline_name, dev_mode)
    logger.info('predicting')
    if chunk_size is not None:
        _predict_in_chunks(pipeline_name, dev_mode, chunk_size)
    else:
        _predict(pipeline_name, dev_mode)


@action.command()
@click.option('-p', '--pipeline_name', help='pipeline to be trained', required=True)
@click.option('-d', '--dev_mode', help='if true only a small sample of data will be used', is_flag=True, required=False)
def train_evaluate(pipeline_name, dev_mode):
    logger.info('training')
    _train(pipeline_name, dev_mode)
    logger.info('evaluate')
    _evaluate(pipeline_name, dev_mode)


if __name__ == "__main__":
    action()
