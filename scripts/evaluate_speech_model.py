#!/usr/bin/env python

from sklearn.linear_model import SGDClassifier

from hrc_speech_prediction.models import get_path_from_cli_arguments
from hrc_speech_prediction.speech_model import SpeechModel
from hrc_speech_prediction.evaluation import Evaluation


N_GRAMS = (1, 2)
TFIDF = False


working_path = get_path_from_cli_arguments()

speech_model_gen = SpeechModel.model_generator(
    SGDClassifier,
    loss='log', average=True, penalty='l2', alpha=.0002)

ev = Evaluation(speech_model_gen, working_path, n_grams=N_GRAMS, tfidf=TFIDF)
ev.evaluate_all()
