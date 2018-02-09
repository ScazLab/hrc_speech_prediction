#!/usr/bin/env python

import os

import numpy as np

from hrc_speech_prediction import defaults
from hrc_speech_prediction.data import (ALL_ACTIONS, TRAIN_PARTICIPANTS,
                                        TrainData)
from hrc_speech_prediction.features import get_bow_features
from hrc_speech_prediction.models import CombinedModel, JointModel


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


def update_speech_for_new_actions(speech_model, vecorizer, weight=1):
    X = vecorizer.transform([
        "white part with red stripes in the middle",
        "white cylindrical base with red lines in the middle",
        "white part with red stripes far apart",
        ("white cylindrical base with one red lines at the top and one "
         "at the bottom"),
    ])
    labels = ['front_2'] * 2 + ['front_4'] * 2
    # Devide weights by two because two utterances for each new action
    weights = np.ones(len(labels)) * weight * .5
    # TODO: the following two lines should be changed
    labels = speech_model._transform_labels(labels)
    speech_model.model.partial_fit(X, labels, sample_weight=weights)


def train_combined_model(speech_eps,
                         context_eps,
                         fit_type="incremental",
                         tfidf=False,
                         n_grams=(1, 2),
                         speech_model_class=JointModel,
                         speech_model_parameters={},
					     init_new_speech_actions=False):

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
        data, tfidf=tfidf, n_grams=n_grams, max_features=None)
    X_speech = X_speech[flat_train_ids, :]

    model_gen = JointModel.model_generator(speech_model_class,
                                           **speech_model_parameters)

    combined_model = CombinedModel(
        vectorizer,
        model_gen,
        ALL_ACTIONS,
        speech_eps=speech_eps,
        context_eps=context_eps)

    if "incremental" in fit_type:
        combined_model.partial_fit(train_context, X_speech, labels)
    elif "offline" in fit_type:
        combined_model.fit(train_context, X_speech, labels)

    if init_new_speech_actions:
        if "incremental" not in fit_type:
            raise NotImplementedError("Can't add speech data on offline speech")
        update_speech_for_new_actions(combined_model.speech_model,
                                      combined_model._vectorizer,
                                      weight=len(labels) * 1. / len(ALL_ACTIONS)
                                      )

    return combined_model
    # model_path = os.path.join(path, 'combined_model_{}{}.pkl'.format(
    #     speech_eps, context_eps))

    # with open(model_path, "wb") as m:
    #     joblib.dump(combined_model, m, compress=9)
    # with open(os.path.join(path, 'vectorizer.pkl'), "wb") as m:
    #     joblib.dump(vectorizer, m, compress=9)
