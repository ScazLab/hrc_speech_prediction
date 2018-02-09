#!/usr/bin/env python


"""Script to evaluate the speech model for a given set of parameters.

- runs cross-validated evaluations on the training set,
- plots the predictions for each utterance in the training set
  to `<working_path>/figs/` (they correspond to training on all
  participants but the one for the utterance),
- plots predictions for some specific example utterances (trained on all
  data).
"""


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
MODEL_PARAMS = {
    'loss': 'log',
    'average': True,
    'penalty': 'l2',
    'alpha': .02,
    'max_iter': 100,
    'tol': 1.e-3,
}

working_path = get_path_from_cli_arguments()
fig_path = os.path.join(working_path, 'figs')
if not os.path.isdir(fig_path):
    os.mkdir(fig_path)

speech_model_gen = JointModel.model_generator(SGDClassifier, **MODEL_PARAMS)

ev = Evaluation(speech_model_gen, working_path, n_grams=N_GRAMS, tfidf=TFIDF,
                model_variations={'speech': {'features': 'speech'}})
ev.evaluate_all()
classes = list(set(ev.data.labels))
utterances = list(ev.data.utterances)
digits = int(np.ceil(np.math.log10(len(utterances))))

plt.set_cmap('Blues')
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


N = 3  # Train N models on all data
models = [
    speech_model_gen(ev.context_actions, features='speech').fit(
        ev.X_context, ev.X_speech, ev.data.labels)
    for _ in range(N)]

TEST_SENTENCES = [
    "",
    "Give me the",
    "Baxter",
    "Please",
    "Get me a green part",
    "Get me a green part with two black stripes at the top",
    "Baxter can you give me a green part with two black stripes at the top",
    "Get me red blue green",
    "Get me red top blue and green at the bottom",
    "Give me the part with the stripes together at the",
    "Give me the part with the stripes equally spread",
    "Give me the cylinder",
    "flat wooden part",
    "Give me the last",
    "Give me the part on the left",
]

X_speech = ev.speech_vectorizer.transform(TEST_SENTENCES)
probas = np.stack([
    m._predict_proba(np.zeros((len(TEST_SENTENCES), len(classes))), X_speech)
    for m in models])

fig, axes = plt.subplots(3, 5, sharey=True)
for i, sentence in enumerate(TEST_SENTENCES):
    plot_predict_proba(probas[:, i, :], classes, sentence, ax=axes[i // 5, i % 5])
plt.show()
