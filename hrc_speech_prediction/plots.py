import numpy as np
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt


def plot_incremental_scores(scores, ax=None, smoothing_window=101, label=None):
    if ax is None:
        ax = plt.gca()
    xaxis = np.arange(len(scores))
    score_smooth = savgol_filter(scores, smoothing_window, 3, mode='mirror')
    plt.scatter(xaxis, scores, marker='x')
    plt.plot(score_smooth, label=label)
