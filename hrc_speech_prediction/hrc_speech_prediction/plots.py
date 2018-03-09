# coding: utf8

from textwrap import wrap

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter


def simplify_plot(ax):
    ax.spines['top'].set_visible(False)
    ax.tick_params(top=False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(right=False)


def get_n_colors(n, color_map=None):
    if color_map is None:
        color_map = plt.get_cmap()
    return [color_map((1. * i) / n) for i in range(n)]


def plot_incremental_scores(scores, ax=None, smoothing_window=101, label=None):
    if ax is None:
        ax = plt.gca()
    xaxis = np.arange(len(scores))
    score_smooth = savgol_filter(scores, smoothing_window, 3, mode='mirror')
    ax.scatter(xaxis, scores, marker='x')
    ax.plot(score_smooth, label=label)


def plot_predict_proba(probas,
                       classes,
                       utterance,
                       model_names=None,
                       truth=None,
                       ax=None,
                       colors=None,
                       color_map=None):
    """Plot predicted probabilities for one or more models.

    :param probas: numpy array, shape is (n_models, n_classes)
    :param classes: list(string)
        names of the classes (in same order as predictions),
    :param utterances: string
    :param model_names: list(string)
    :param truth: string
        ground truth label
    """
    if ax is None:
        ax = plt.gca()
    if probas.shape < 2:
        probas = probas[None, :]
    n_models = probas.shape[0]
    shift = .8 / n_models
    if colors is None:
        colors = get_n_colors(n_models, color_map=color_map)
    xs = np.arange(probas.shape[1])
    for i in range(n_models):
        rects = ax.bar(
            xs + i * shift, probas[i, :], width=shift, color=colors[i]).patches
        # Draw * above highest prediction
        best = rects[np.argmax(probas[i, :])]
        ax.text(
            best.get_x() + best.get_width() / 2,
            best.get_height() * 1.01,
            '*',
            ha='center',
            va='bottom',
            fontsize="12")
    ax.set_xticks(.8 + xs)
    ticks = ax.set_xticklabels(classes, rotation=70, ha='right')
    if truth is not None:
        ticks[classes.index(truth)].set_weight('black')

    ax.set_xlim(0, len(classes))
    ax.set_title("\n".join(wrap(u'“' + utterance + u'”', 70)), fontsize="11")
    simplify_plot(ax)

    plt.xlabel("Actions")
    plt.ylabel("Probability")
    if model_names is not None:
        ax.legend(model_names)
