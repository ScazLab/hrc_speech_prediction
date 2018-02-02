from textwrap import wrap

import numpy as np
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt


def get_n_colors(n, color_map=plt.get_cmap()):
    return [color_map((1. * i) / n) for i in range(n)]


def plot_incremental_scores(scores, ax=None, smoothing_window=101, label=None):
    if ax is None:
        ax = plt.gca()
    xaxis = np.arange(len(scores))
    score_smooth = savgol_filter(scores, smoothing_window, 3, mode='mirror')
    plt.scatter(xaxis, scores, marker='x')
    plt.plot(score_smooth, label=label)


def plot_predict_proba(probas, classes, utterance,
                       model_names=None, truth=None, ax=None):
    if ax is None:
        ax = plt.gca()
    if model_names is None:
        model_names = [None]
        width = .8
        shift = 0.
    else:
        width = 1. / (1 + len(model_names))
        shift = 1. / len(model_names)
    colors = get_n_colors(len(model_names))
    xs = np.arange(probas.shape[1])
    for i, _ in enumerate(model_names):
        rects = ax.bar(xs + i * shift, probas[i, :], width=width,
                       color=colors[i]).patches
        # Draw * above highest prediction
        best = rects[np.argmax(probas[i, :])]
        ax.text(best.get_x() + best.get_width() / 2,
                best.get_height() * 1.01, '*', ha='center',
                va='bottom', fontsize="12")
    _, ticks = plt.xticks(xs, classes, rotation=70)
    if truth is not None:
        ticks[classes.index(truth)].set_weight('black')
    ax.set_title("\n".join(wrap(utterance, 100)), fontsize="9")
    if model_names != [None]:
        ax.legend(model_names)
