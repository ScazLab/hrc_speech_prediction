#!/usr/bin/env python


import os
import argparse

import numpy as np
from scipy import sparse
from sklearn.externals import joblib
from sklearn.linear_model import LogisticRegression, SGDClassifier

from hrc_speech_prediction.data import TrainData, TRAIN_PARTICIPANTS, ALL_ACTIONS
from hrc_speech_prediction.features import (get_context_features,
                                            get_bow_features)
from hrc_speech_prediction.models import ContextFilterModel
from hrc_speech_prediction import combined_model as cm


TFIDF = False
N_GRAMS = (1, 2)
ADD_LAST_ACTION = True

parser = argparse.ArgumentParser("Train and evaluate classifier")
parser.add_argument('path', help='path to the experiment data',
                    default=os.path.curdir)
args = parser.parse_args()

data = TrainData.load(os.path.join(args.path, "train.json"))


flat_train_ids = [i for p in TRAIN_PARTICIPANTS for i in data.data[p].ids]
train_ids_by_trial = [
    trial.ids for p in TRAIN_PARTICIPANTS
    for trial in data.data[p]
]


# Get features
X_context, _ = get_context_features(data, actions=ALL_ACTIONS)
X_context = X_context[flat_train_ids, :]
X_speech, vectorizer = get_bow_features(data, tfidf=TFIDF, n_grams=N_GRAMS,
                                        max_features=None)
X_speech = X_speech[flat_train_ids, :]
labels = [list(data.labels)[i] for i in flat_train_ids]
if ADD_LAST_ACTION:
    X_s_new = vectorizer.transform([
        "white part with red stripes in the middle",
        "white cylindrical base with red lines in the middle",
        "white part with red stripes far apart",
        "white cylindrical base with one red lines at the top and one at the bottom",
    ])
    X_speech = sparse.vstack((X_speech, X_s_new))
    # use neutral context for last actions
    X_c_new = np.tile(np.average(X_context, axis=0), (4, 1))
    X_context = np.concatenate((X_context, X_c_new), axis=0)
    labels = labels + ['front_2'] * 2 + ['front_4'] * 2
    weights = np.ones(len(labels))
    weights[-4:] = len(labels) * 1. / len(ALL_ACTIONS)
else:
    weights = None

features = "speech"
speech_eps = 0.15
context_eps = 0.15


model = ContextFilterModel.model_generator(SGDClassifier,
                                                  loss='log',
                                                  average=True,
                                                  penalty='l2')
speech_model = model(ALL_ACTIONS,
                     features=features,
                     randomize_context=.3,).fit(X_context,
                                                X_speech,
                                                labels,
                                                sample_weight=weights)

combined_model = cm.CombinedModel(root=cm.Node(),
                                  speech_eps=speech_eps,
                                  context_eps=context_eps,
                                  speech_model=speech_model,
                                  vectorizer=vectorizer)

for t in train_ids_by_trial:
    combined_model.add_branch(t)

model_path = os.path.join(args.path, 'combined_model_{}{}.pkl'.format(
    speech_eps, context_eps))

with open(model_path, "wb") as m:
    joblib.dump(speech_model, m, compress=9)
with open(os.path.join(args.path, 'vectorizer.pkl'), "wb") as m:
    joblib.dump(vectorizer, m, compress=9)
