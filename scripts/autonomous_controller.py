#!/usr/bin/env python

import rospy
import numpy as np
from sklearn.externals import joblib

from human_robot_collaboration.controller import BaseController


class DummyPredictor(object):

    def __init__(self, object_list):
        self.obj = object_list
        self.words = [o.split('_')[0] for o in self.obj]

    @property
    def n_obj(self):
        return len(self.obj)

    def transform(self, utterances):
        return np.array([[w in u.lower() for w in self.words]
                         for u in utterances])

    def predict(self, Xs, Xc):
        # return an object that is in context and which name is in utterance
        X = np.concatenate(Xs, Xc, axis=1)
        intersection = X[:, :self.n_obj] * X[:, self.n_obj:]
        chosen = -np.ones((X.shape[0]), dtype='int8')
        ii, jj = intersection.nonzero()
        chosen[ii] = jj
        scr = self.obj.index('screwdriver_1')
        chosen[(chosen == -1).nonzero()[0]] = scr
        return [self.obj[c] for c in chosen]


class SpeechPredictionController(BaseController):

    OBJECT_DICT = {
        "seat":          (BaseController.LEFT, 200),
        "chair_back":    (BaseController.LEFT, 198),
        "leg_1":         (BaseController.LEFT, 150),
        "leg_2":         (BaseController.LEFT, 151),
        "leg_3":         (BaseController.LEFT, 152),
        "leg_4":         (BaseController.LEFT, 153),
        "leg_5":         (BaseController.LEFT, 154),
        "leg_6":         (BaseController.LEFT, 155),
        "leg_7":         (BaseController.LEFT, 156),
        "foot_1":        (BaseController.RIGHT, 10),
        "foot_2":        (BaseController.RIGHT, 11),
        "foot_3":        (BaseController.RIGHT, 12),
        "foot_4":        (BaseController.RIGHT, 13),
        "front_1":       (BaseController.RIGHT, 14),
        "front_2":       (BaseController.RIGHT, 15),
        "top_1":         (BaseController.RIGHT, 16),
        "top_2":         (BaseController.RIGHT, 17),
        "back_1":        (BaseController.RIGHT, 18),
        "back_2":        (BaseController.RIGHT, 19),
        "screwdriver_1": (BaseController.RIGHT, 20),
        "front_3":       (BaseController.RIGHT, 22),
        "front_4":       (BaseController.RIGHT, 23),
    }
    BRING = 'get_pass'

    def __init__(self, timer_path=None, debug=False):
        super(SpeechPredictionController, self).__init__(
            left=True, right=True, speech=False, listen=True, recovery=True)
        if debug:
            self.model = DummyPredictor(list(self.OBJECT_DICT.keys()))
            self.vectorizer = self.model
            self.actions_in_context = self.model.obj
        else:
            model_path = rospy.get_param('/speech_prediction/model_path')
            self.model = joblib.load(model_path)
            # utterance vectorizer
            vectorizer_path = rospy.get_param('/speech_prediction/vectorizer_path')
            self.vectorizer = joblib.load(vectorizer_path)
            # actions in order of context vector
            self.actions_in_context = self.model.actions

    def _run(self):
        rospy.loginfo('Starting autonomous control')
        self.context = np.ones((len(self.actions_in_context)), dtype='bool')
        while not self.finished:
            utterance = None
            while not utterance:
                rospy.loginfo('Waiting for utterance')
                utterance = self.listen_sub.wait_for_msg(timeout=60.)
                rospy.loginfo('found: {}'.format(utterance))
            x_u = self.vectorizer.transform([utterance])
            action = self.model.predict(x_u, self.context[None, :])[0]
            rospy.loginfo("Taking action {} for \"{}\"".format(action, utterance))
            self.take_action(action)

    def take_action(self, action):
        side, obj = self.OBJECT_DICT[action]
        return self._action(side, (self.BRING, [obj]), {'wait': True})

    def _update_context(self, action):
        self.context[self.actions_in_context.index(action)] = 0


# TODO: move to ros parameters
timer_path = '/tmp/timer.json'
controller = SpeechPredictionController(timer_path=timer_path, debug=True)

controller.run()
