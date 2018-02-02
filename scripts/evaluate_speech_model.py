#!/usr/bin/env python

import os

import numpy as np
from sklearn.linear_model import SGDClassifier
from matplotlib import pyplot as plt

from hrc_speech_prediction.models import (
    JointModel, get_path_from_cli_arguments)
from hrc_speech_prediction.evaluation import Evaluation, TRAIN_PARTICIPANTS
from hrc_speech_prediction.plots import plot_predict_proba


N_GRAMS = (1, 2)
TFIDF = False


working_path = get_path_from_cli_arguments()
fig_path = os.path.join(working_path, 'figs')
if not os.path.isdir(fig_path):
    os.mkdir(fig_path)

speech_model_gen = JointModel.model_generator(
    SGDClassifier,
    loss='log', average=True, penalty='l2', alpha=.0002)

ev = Evaluation(speech_model_gen, working_path, n_grams=N_GRAMS, tfidf=TFIDF,
                model_variations={'speech': {'features': 'speech'}})
ev.evaluate_all()
classes = list(set(ev.data.labels))
utterances = list(ev.data.utterances)
digits = int(np.ceil(np.math.log10(len(utterances))))

fig = plt.figure()
for tst in TRAIN_PARTICIPANTS:
    train_idx = [i for p in TRAIN_PARTICIPANTS
                 for i in list(ev.data.data[p].ids)
                 if not p == tst]
    X_train = ev.get_Xs(train_idx)
    model = speech_model_gen(ev.context_actions, features='speech').fit(
        X_train[0], X_train[1], ev.get_labels(train_idx))
    # Evaluation
    test_idx = list(ev.data.data[tst].ids)
    ev.check_indices(train_idx, test_idx)
    probas = model._predict_proba(*ev.get_Xs(test_idx))
    labels = ev.get_labels(train_idx)
    for i, p in enumerate(probas):
        plot_predict_proba(p[None, :], classes, utterances[test_idx[i]],
                           truth=labels[i])
        fig.tight_layout()
        plt.savefig(os.path.join(fig_path,
                    "fig.{:0{digits}d}.png".format(test_idx[i], digits=digits)))
        fig.clf()
