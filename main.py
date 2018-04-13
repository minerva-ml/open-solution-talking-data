import os
import shutil

import click
import pandas as pd
from deepsense import neptune
from sklearn.metrics import roc_auc_score

from pipeline_config import SOLUTION_CONFIG, FEATURE_COLUMNS, TARGET_COLUMNS, CV_COLUMNS
from pipelines import PIPELINES
from utils import init_logger, read_params, create_submission, set_seed, train_valid_split_on_timestamp, \
    read_csv_last_n_rows

set_seed(1234)
logger = init_logger()
ctx = neptune.Context()
params = read_params(ctx)


@click.group()
def action():
    pass


@action.command()
@click.option('-p', '--pipeline_name', help='pipeline to be trained', required=True)
@click.option('-v', '--validation_size', help='percentage of training used for validation', default=0.1, required=False)
@click.option('-n', '--read_n_rows', help='read first n rows of data', default=20000000, required=False)
@click.option('-d', '--dev_mode', help='if true only a small sample of data will be used', is_flag=True, required=False)
def train(pipeline_name, validation_size, read_n_rows, dev_mode):
    _train(pipeline_name, validation_size, read_n_rows, dev_mode)


def _train(pipeline_name, validation_size, read_n_rows, dev_mode):
    if bool(params.overwrite) and os.path.isdir(params.experiment_dir):
        shutil.rmtree(params.experiment_dir)

    if dev_mode:
        meta_train = read_csv_last_n_rows(os.path.join(params.data_dir, 'train.csv'), nrows=read_n_rows)
    else:
        meta_train = pd.read_csv(os.path.join(params.data_dir, 'train.csv'), nrows=10000)

    meta_train_split, meta_valid_split = train_valid_split_on_timestamp(meta_train, validation_size,
                                                                        timestamp_column=CV_COLUMNS)

    if dev_mode:
        meta_train_split = meta_train_split.sample(100)
        meta_valid_split = meta_valid_split.sample(10)

    data = {'input': {'X': meta_train_split[FEATURE_COLUMNS],
                      'y': meta_train_split[TARGET_COLUMNS],
                      'X_valid': meta_valid_split[FEATURE_COLUMNS],
                      'y_valid': meta_valid_split[TARGET_COLUMNS],
                      },
            }

    pipeline = PIPELINES[pipeline_name]['train'](SOLUTION_CONFIG)
    pipeline.clean_cache()
    pipeline.fit_transform(data)
    pipeline.clean_cache()


@action.command()
@click.option('-p', '--pipeline_name', help='pipeline to be trained', required=True)
@click.option('-v', '--validation_size', help='percentage of training used for validation', default=0.1, required=False)
@click.option('-n', '--read_n_rows', help='read first n rows of data', default=20000000, required=False)
@click.option('-d', '--dev_mode', help='if true only a small sample of data will be used', is_flag=True, required=False)
def evaluate(pipeline_name, validation_size, read_n_rows, dev_mode):
    _evaluate(pipeline_name, validation_size, read_n_rows, dev_mode)


def _evaluate(pipeline_name, validation_size, read_n_rows, dev_mode):
    if dev_mode:
        meta_train = read_csv_last_n_rows(os.path.join(params.data_dir, 'train.csv'), nrows=read_n_rows)
    else:
        meta_train = pd.read_csv(os.path.join(params.data_dir, 'train.csv'), nrows=1000)

    meta_train_split, meta_valid_split = train_valid_split_on_timestamp(meta_train, validation_size,
                                                                        timestamp_column=CV_COLUMNS)

    if dev_mode:
        meta_valid_split = meta_valid_split.sample(10)

    data = {'input': {'X': meta_valid_split[FEATURE_COLUMNS],
                      'y': None,
                      },
            }
    y_true = meta_valid_split[TARGET_COLUMNS].values

    pipeline = PIPELINES[pipeline_name]['inference'](SOLUTION_CONFIG)
    pipeline.clean_cache()
    output = pipeline.transform(data)
    pipeline.clean_cache()
    y_pred = output['y_pred']

    logger.info('Calculating ROC_AUC Scores')
    score = roc_auc_score(y_true, y_pred)
    logger.info('ROC_AUC score on validation is {}'.format(score))
    ctx.channel_send('ROC_AUC', 0, score)


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
    if dev_mode:
        meta_test = pd.read_csv(os.path.join(params.data_dir, 'test.csv'), nrows=10)
    else:
        meta_test = pd.read_csv(os.path.join(params.data_dir, 'test.csv'))

    data = {'input': {'X': meta_test[FEATURE_COLUMNS],
                      'y': None,
                      },
            }

    pipeline = PIPELINES[pipeline_name]['inference'](SOLUTION_CONFIG)
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
    test_filename = os.path.join(params.data_dir, 'test.csv')

    submission_chunks = []
    for meta_test_chunk in pd.read_csv(test_filename, chunksize=chunk_size):
        data = {'input': {'meta': meta_test_chunk,
                          },
                }

        pipeline = PIPELINES[pipeline_name]['inference'](SOLUTION_CONFIG)
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
@click.option('-v', '--validation_size', help='percentage of training used for validation', default=0.1, required=False)
@click.option('-n', '--read_n_rows', help='read first n rows of data', default=20000000, required=False)
@click.option('-d', '--dev_mode', help='if true only a small sample of data will be used', is_flag=True, required=False)
@click.option('-c', '--chunk_size', help='size of the chunks to run prediction on', type=int, default=None,
              required=False)
def train_evaluate_predict(pipeline_name, validation_size, read_n_rows, dev_mode, chunk_size):
    logger.info('training')
    train(pipeline_name, validation_size, read_n_rows, dev_mode)
    logger.info('evaluate')
    evaluate(pipeline_name, validation_size, read_n_rows, dev_mode)
    logger.info('predicting')
    predict(pipeline_name, dev_mode, chunk_size)


@action.command()
@click.option('-p', '--pipeline_name', help='pipeline to be trained', required=True)
@click.option('-v', '--validation_size', help='percentage of training used for validation', default=0.1, required=False)
@click.option('-n', '--read_n_rows', help='read first n rows of data', default=20000000, required=False)
@click.option('-d', '--dev_mode', help='if true only a small sample of data will be used', is_flag=True, required=False)
@click.option('-c', '--chunk_size', help='size of the chunks to run prediction on', type=int, default=None,
              required=False)
def evaluate_predict(pipeline_name, validation_size, read_n_rows, dev_mode, chunk_size):
    logger.info('evaluate')
    evaluate(pipeline_name, validation_size, read_n_rows, dev_mode)
    logger.info('predicting')
    predict(pipeline_name, dev_mode, chunk_size)


@action.command()
@click.option('-p', '--pipeline_name', help='pipeline to be trained', required=True)
@click.option('-v', '--validation_size', help='percentage of training used for validation', default=0.1, required=False)
@click.option('-n', '--read_n_rows', help='read first n rows of data', default=20000000, required=False)
@click.option('-d', '--dev_mode', help='if true only a small sample of data will be used', is_flag=True, required=False)
def train_evaluate(pipeline_name, validation_size, read_n_rows, dev_mode):
    logger.info('training')
    train(pipeline_name, validation_size, read_n_rows, dev_mode)
    logger.info('evaluate')
    evaluate(pipeline_name, validation_size, read_n_rows, dev_mode)


if __name__ == "__main__":
    action()
