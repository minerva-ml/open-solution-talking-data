from functools import partial

from sklearn.ensemble import RandomForestClassifier

from steps.base import Step, Dummy, sparse_hstack_inputs
from steps.sklearn.models import make_transformer


def simple_pipe(config, train_mode=True):
    tfidf_char_vectorizer, tfidf_word_vectorizer = _tfidf(preprocessed_input, config)

    tfidf_logreg = Step(name='tfidf_logreg',
                        transformer=LogisticRegressionMultilabel(**config.logistic_regression_multilabel),
                        input_steps=[preprocessed_input, tfidf_char_vectorizer, tfidf_word_vectorizer],
                        adapter={'X': ([('tfidf_char_vectorizer', 'features'),
                                        ('tfidf_word_vectorizer', 'features')], sparse_hstack_inputs),
                                 'y': ([('cleaning_output', 'y')]),
                                 },
                        cache_dirpath=config.env.cache_dirpath)
    output = Step(name='tfidf_logreg_output',
                  transformer=Dummy(),
                  input_steps=[tfidf_logreg],
                  adapter={'y_pred': ([('tfidf_logreg', 'prediction_probability')]),
                           },
                  cache_dirpath=config.env.cache_dirpath)
    return output


def extract_features_app(config):
    return


PIPELINES = {'dummy_pipe': {'train': partial(simple_pipe, train_mode=True),
                            'inference': partial(simple_pipe, train_mode=False)},
             }
