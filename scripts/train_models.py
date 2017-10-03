#!/usr/bin/env python


import os
import argparse

from sklearn.externals import joblib
from sklearn.linear_model import LogisticRegression

from hrc_speech_prediction.data import TrainData, TRAIN_PARTICIPANTS, ALL_ACTIONS
from hrc_speech_prediction.features import (get_context_features,
                                            get_bow_features)
from hrc_speech_prediction.models import ContextFilterModel


TFIDF = False
N_GRAMS = (1, 2)
ADD_LAST_ACTION = False

parser = argparse.ArgumentParser("Train and evaluate classifier")
parser.add_argument('path', help='path to the experiment data',
                    default=os.path.curdir)
args = parser.parse_args()

data = TrainData.load(os.path.join(args.path, "train.json"))
train_ids = [i for p in TRAIN_PARTICIPANTS for i in data.data[p].ids]
if ADD_LAST_ACTION:
    # Adds one data point for missing objects
    last_action_ids = [data.data['15.ADT'][-1].ids[i] for i in (-4, -2)]
    assert([data.get_pair_from_id(i)[0].label for i in last_action_ids
            ] == ['front_2', 'front_4'])
    train_ids.extend(last_action_ids)
# Get features
X_context, _ = get_context_features(data, actions=ALL_ACTIONS)
X_context = X_context[train_ids, :]
if ADD_LAST_ACTION:
    # use neutral context for last actions
    X_context[-2:, :] = .5
X_speech, vocabulary = get_bow_features(data, tfidf=TFIDF, n_grams=N_GRAMS,
                                        max_features=None)
X_speech = X_speech[train_ids, :]
labels = [list(data.labels)[i] for i in train_ids]

for features in ('speech', 'both'):
    model = ContextFilterModel(
        LogisticRegression(), ALL_ACTIONS, features=features
    ).fit(X_context, X_speech, labels)
    model_path = os.path.join(args.path, 'model_{}.pkl'.format(features))
    with open(model_path, "wb") as m:
        joblib.dump(model, m, compress=9)
with open(os.path.join(args.path, 'vocabulary.pkl'), "wb") as m:
    joblib.dump(vocabulary, m, compress=9)
