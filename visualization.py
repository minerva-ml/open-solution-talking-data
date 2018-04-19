import io
import numpy as np

import PIL
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import roc_curve, roc_auc_score


def roc_curve_plot(y_true, y_pred, figsize=(12, 8)):
    auc = roc_auc_score(y_true, y_pred)
    fpr, tpr, thresholds = roc_curve(y_true, y_pred, pos_label=0)

    fig, ax = plt.subplots(1, figsize=figsize)

    ax.set_title('AUC: {}'.format(auc))
    ax.plot(tpr, fpr)
    ax.plot([0, 1], [0, 1], 'r--')

    return fig


def prediction_distribution_plot(y_true, y_pred, sample=None, bins=20, figsize=(16, 12)):
    if sample is not None:
        y_pred = np.random.choice(y_pred, sample, replace=False)

    mean_df = pd.DataFrame({'mean_true': [y_true.mean()], 'mean_pred': [y_pred.mean()]})

    y_pred_neg = y_pred[np.where(y_pred < 0.5)]
    y_pred_pos = y_pred[np.where(y_pred >= 0.5)]

    fig, axs = plt.subplots(2, 2, figsize=figsize)

    mean_df.plot(kind='bar', ax=axs[0, 0])

    axs[0, 1].hist(y_pred, range=(0, 1), bins=bins, label='positive', histtype="step", lw=2)
    axs[0, 1].axvline(x=0.5, color='r', linestyle='--')

    axs[1, 0].hist(y_pred_neg, range=(0, 0.5), bins=bins, label='positive', histtype="step", lw=2)
    axs[1, 1].hist(y_pred_pos, range=(0.5, 1), bins=bins, label='positive', histtype="step", lw=2)

    return fig


def fig_to_pil(fig):
    buffer = io.BytesIO()
    fig.savefig(buffer, format='png')
    buffer.seek(0)
    img = PIL.Image.open(buffer)
    buffer.close()
    return img
