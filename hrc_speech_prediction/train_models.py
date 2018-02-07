#!/usr/bin/env python

import os

from sklearn.linear_model import SGDClassifier

from hrc_speech_prediction import defaults
from hrc_speech_prediction.data import (ALL_ACTIONS, TRAIN_PARTICIPANTS,
                                        TrainData)
from hrc_speech_prediction.features import get_bow_features
from hrc_speech_prediction.models import CombinedModel
from hrc_speech_prediction.speech_model import SpeechModel


def get_labels(indices, data):
    return [list(data.labels)[i] for i in indices]


def format_cntxt_indices(data, indices):
    cntxts = []
    actions = []

    all_labels = list(data.labels)
    for t in indices:
        cntxt = []
        for i in t:
            label = all_labels[i]
            cntxts.append([c for c in cntxt])
            actions.append(label)

            cntxt.append(label)

    return cntxts, actions


def train_combined_model(speech_eps, context_eps, fit_type="incremental"):
    TFIDF = False
    N_GRAMS = (1, 2)
    alpha = .04

    path = defaults.DATA_PATH
    print("PATH: ", os.path.join(path, "train.json"))

    data = TrainData.load(os.path.join(path, "train.json"))

    flat_train_ids = [i for p in TRAIN_PARTICIPANTS for i in data.data[p].ids]
    train_ids_by_trial = [
        trial.ids for p in TRAIN_PARTICIPANTS for trial in data.data[p]
    ]

    # Get features
    train_context, labels = format_cntxt_indices(data, train_ids_by_trial)
    X_speech, vectorizer = get_bow_features(
        data, tfidf=TFIDF, n_grams=N_GRAMS, max_features=None)
    X_speech = X_speech[flat_train_ids, :]

    model_gen = SpeechModel.model_generator(
        SGDClassifier, loss='log', average=True, penalty='l2', alpha=alpha)

    combined_model = CombinedModel(
        vectorizer=vectorizer,
        model_generator=model_gen,
        actions=ALL_ACTIONS,
        speech_eps=speech_eps,
        context_eps=context_eps)

    if "incremental" in fit_type:
        combined_model.partial_fit(train_context, X_speech, labels)
    elif "batch" in fit_type:
        combined_model.fit(train_context, X_speech, labels)

    return combined_model
    # model_path = os.path.join(path, 'combined_model_{}{}.pkl'.format(
    #     speech_eps, context_eps))

    # with open(model_path, "wb") as m:
    #     joblib.dump(combined_model, m, compress=9)
    # with open(os.path.join(path, 'vectorizer.pkl'), "wb") as m:
    #     joblib.dump(vectorizer, m, compress=9)
