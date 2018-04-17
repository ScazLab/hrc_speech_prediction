#!/usr/bin/env python

from matplotlib import pyplot as plt
from sklearn.linear_model import SGDClassifier

from hrc_speech_prediction.evaluation import Evaluation
from hrc_speech_prediction.models import (JointModel,
                                          get_path_from_cli_arguments)
from hrc_speech_prediction.plots import plot_incremental_scores

N_GRAMS = (1, 3)
TFIDF = False

working_path = get_path_from_cli_arguments()

speech_model_gen = JointModel.model_generator(
    SGDClassifier, loss='log', average=True, penalty='l2', alpha=.04)

ev = Evaluation(
    speech_model_gen,
    working_path,
    n_grams=N_GRAMS,
    tfidf=TFIDF,
    model_variations={
        k: {
            'features': k
        }
        for k in ['speech', 'context', 'both']
    })
ev.evaluate_all()
scores = ev.evaluate_incremental_learning(shuffle=False)
for m in scores:
    plot_incremental_scores(scores[m], label=m)
plt.legend()
plt.show()
