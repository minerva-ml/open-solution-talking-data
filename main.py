import os
import shutil

import click
import pandas as pd
from deepsense import neptune
from sklearn.metrics import roc_auc_score

import pipeline_config as cfg
from pipelines import PIPELINES
from utils import init_logger, read_params, create_submission, set_seed, save_evaluation_predictions, \
    read_csv_time_chunks, cut_data_in_time_chunks, data_hash_channel_send, get_submission_hours_index

set_seed(1234)
logger = init_logger()
ctx = neptune.Context()
params = read_params(ctx)


@click.group()
def action():
    pass


@action.command()
def prepare_data():
    logger.info('chunking train')
    train = pd.read_csv(params.raw_train_filepath)
    cut_data_in_time_chunks(train,
                            timestamp_column='click_time',
                            chunks_dir=params.train_chunks_dir,
                            prefix='train',
                            logger=logger)

    logger.info('chunking test')
    test = pd.read_csv(params.test_suplement_filepath)
    cut_data_in_time_chunks(test,
                            timestamp_column='click_time',
                            chunks_dir=params.test_chunks_dir,
                            prefix='test',
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
        TRAIN_DAYS_HOURS = cfg.DEV_TRAIN_DAYS_HOURS
        VALID_DAYS_HOURS = cfg.DEV_VALID_DAYS_HOURS
    else:
        TRAIN_DAYS_HOURS = eval(params.train_days_hours)
        VALID_DAYS_HOURS = eval(params.valid_days_hours)

    meta_train_split = read_csv_time_chunks(params.train_chunks_dir,
                                            prefix='train',
                                            days_hours=TRAIN_DAYS_HOURS,
                                            usecols=cfg.FEATURE_COLUMNS + cfg.TARGET_COLUMNS,
                                            dtype=cfg.COLUMN_TYPES['train'],
                                            logger=logger)
    meta_valid_split = read_csv_time_chunks(params.train_chunks_dir,
                                            prefix='train',
                                            days_hours=VALID_DAYS_HOURS,
                                            usecols=cfg.FEATURE_COLUMNS + cfg.TARGET_COLUMNS,
                                            dtype=cfg.COLUMN_TYPES['train'],
                                            logger=logger)

    data_hash_channel_send(ctx, 'Training Data Hash', meta_train_split)
    data_hash_channel_send(ctx, 'Validation Data Hash', meta_valid_split)

    if dev_mode:
        meta_train_split = meta_train_split.sample(cfg.DEV_SAMPLE_TRAIN_SIZE, replace=False)
        meta_valid_split = meta_valid_split.sample(cfg.DEV_SAMPLE_VALID_SIZE, replace=False)

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
        VALID_DAYS_HOURS = cfg.DEV_VALID_DAYS_HOURS
    else:
        VALID_DAYS_HOURS = eval(params.valid_days_hours)

    meta_valid_split = read_csv_time_chunks(params.train_chunks_dir,
                                            prefix='train',
                                            days_hours=VALID_DAYS_HOURS,
                                            usecols=cfg.FEATURE_COLUMNS + cfg.TARGET_COLUMNS,
                                            dtype=cfg.COLUMN_TYPES['train'],
                                            logger=logger)

    data_hash_channel_send(ctx, 'Evaluation Data Hash', meta_valid_split)

    if dev_mode:
        meta_valid_split = meta_valid_split.sample(cfg.DEV_SAMPLE_VALID_SIZE, replace=False)

    logger.info('Target distribution in valid: {}'.format(meta_valid_split['is_attributed'].mean()))

    data = {'input': {'X': meta_valid_split[cfg.FEATURE_COLUMNS],
                      'y': None,
                      },
            }
    pipeline = PIPELINES[pipeline_name]['inference'](cfg.SOLUTION_CONFIG)
    pipeline.clean_cache()
    output = pipeline.transform(data)
    pipeline.clean_cache()
    y_pred = output['y_pred']
    y_true = meta_valid_split[cfg.TARGET_COLUMNS].values.reshape(-1)

    logger.info('Saving evaluation predictions')
    save_evaluation_predictions(params.experiment_dir, y_true, y_pred, meta_valid_split)

    logger.info('Calculating ROC_AUC Full Scores')
    score = roc_auc_score(y_true, y_pred)
    logger.info('ROC_AUC score on full_validation is {}'.format(score))
    ctx.channel_send('ROC_AUC FULL', 0, score)

    logger.info('Subsetting on submission times')
    index_for_submission_hours = get_submission_hours_index(meta_valid_split,
                                                            cfg.TIMESTAMP_COLUMN,
                                                            eval(params.submission_hours))
    y_pred_ = y_pred[index_for_submission_hours]
    y_true_ = y_true[index_for_submission_hours]

    logger.info('Calculating ROC_AUC Submission Scores')
    score = roc_auc_score(y_true_, y_pred_)
    logger.info('ROC_AUC score on submission subset of validation is {}'.format(score))
    ctx.channel_send('ROC_AUC SUBSET', 0, score)


@action.command()
@click.option('-p', '--pipeline_name', help='pipeline to be trained', required=True)
@click.option('-d', '--dev_mode', help='if true only a small sample of data will be used', is_flag=True, required=False)
def predict(pipeline_name, dev_mode):
    _predict(pipeline_name, dev_mode)


def _predict(pipeline_name, dev_mode):
    logger.info('reading data in')
    if dev_mode:
        TEST_DAYS_HOURS = cfg.DEV_TEST_DAYS_HOURS
    else:
        TEST_DAYS_HOURS = eval(params.test_days_hours)

    meta_test_suplement = read_csv_time_chunks(params.test_chunks_dir,
                                               prefix='test',
                                               days_hours=TEST_DAYS_HOURS,
                                               usecols=cfg.FEATURE_COLUMNS + cfg.ID_COLUMN,
                                               dtype=cfg.COLUMN_TYPES['inference'],
                                               logger=logger)
    meta_test = pd.read_csv(params.test_filepath,
                            usecols=cfg.FEATURE_COLUMNS + cfg.ID_COLUMN,
                            dtype=cfg.COLUMN_TYPES['inference'])
    meta_test_full = pd.concat([meta_test_suplement, meta_test], axis=0).reset_index(drop=True)
    meta_test_full.drop_duplicates(subset=cfg.ID_COLUMN, keep='last', inplace=True)
    meta_test_full['click_time'] = pd.to_datetime(meta_test_full['click_time'], format='%Y-%m-%d %H:%M:%S')

    data_hash_channel_send(ctx, 'Test Data Hash', meta_test_full)

    if dev_mode:
        meta_test_full = meta_test_full.sample(cfg.DEV_SAMPLE_TEST_SIZE, replace=False)

    data = {'input': {'X': meta_test_full[cfg.FEATURE_COLUMNS],
                      'y': None,
                      },
            }

    pipeline = PIPELINES[pipeline_name]['inference'](cfg.SOLUTION_CONFIG)
    pipeline.clean_cache()
    output = pipeline.transform(data)
    pipeline.clean_cache()
    y_pred = output['y_pred']

    logger.info('creating submission full test')
    full_submission = create_submission(meta_test_full, y_pred)
    full_submission_filepath = os.path.join(params.experiment_dir, 'full_submission.csv')
    full_submission.to_csv(full_submission_filepath, index=None, encoding='utf-8')

    logger.info('subsetting submission')
    submission = pd.merge(full_submission, meta_test[cfg.ID_COLUMN], on=cfg.ID_COLUMN, how='inner')

    submission_filepath = os.path.join(params.experiment_dir, 'submission.csv')
    submission.to_csv(submission_filepath, index=None, encoding='utf-8')
    logger.info('submission saved to {}'.format(submission_filepath))
    logger.info('submission head \n\n{}'.format(submission.head()))


@action.command()
@click.option('-p', '--pipeline_name', help='pipeline to be trained', required=True)
@click.option('-d', '--dev_mode', help='if true only a small sample of data will be used', is_flag=True, required=False)
def train_evaluate_predict(pipeline_name, dev_mode):
    logger.info('TRAINING')
    _train(pipeline_name, dev_mode)
    logger.info('EVALUATION')
    _evaluate(pipeline_name, dev_mode)
    logger.info('PREDICTION')
    _predict(pipeline_name, dev_mode)


@action.command()
@click.option('-p', '--pipeline_name', help='pipeline to be trained', required=True)
@click.option('-d', '--dev_mode', help='if true only a small sample of data will be used', is_flag=True, required=False)
def evaluate_predict(pipeline_name, dev_mode):
    logger.info('EVALUATION')
    _evaluate(pipeline_name, dev_mode)
    logger.info('PREDICTION')
    _predict(pipeline_name, dev_mode)


@action.command()
@click.option('-p', '--pipeline_name', help='pipeline to be trained', required=True)
@click.option('-d', '--dev_mode', help='if true only a small sample of data will be used', is_flag=True, required=False)
def train_evaluate(pipeline_name, dev_mode):
    logger.info('TRAINING')
    _train(pipeline_name, dev_mode)
    logger.info('EVALUATION')
    _evaluate(pipeline_name, dev_mode)


if __name__ == "__main__":
    action()
