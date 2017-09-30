#!/usr/bin/env python

import rospy
import numpy as np
from sklearn.externals import joblib

from human_robot_collaboration.controller import BaseController


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

    def __init__(self, timer_path=None):
        super(SpeechPredictionController, self).__init__(
            left=True, right=True, speech=False, listen=True, recovery=True)
        model_path = rospy.get_param('/speech_prediction/model_path')
        self.model = joblib.load(model_path)
        # utterance vectorizer
        vectorizer_path = rospy.get_param('/speech_prediction/vectorizer_path')
        self.vectorizer = joblib.load(vectorizer_path)
        # actions in order of context vector
        context_state_path = rospy.get_param('/speech_prediction/context_path')
        self.actions_in_context = joblib.load(context_state_path)

    def _run(self):
        self.context = np.zeros((len(self.actions_in_context)), dtype='bool')
        while not self.finished:
            utterance = self.listen_sub.wait_for_msg()
            x_u = self.vectorizer.transform([utterance])
            action = self.model.predict(np.hstack([x_u[None, :], self.context]))
            rospy.loginfo("Taking action {} for \"{}\"".format(action, utterance))
            self.take_action(self, action)
            rospy.spin()

    def take_action(self, action):
        side, obj = self.OBJECT_DICT[action]
        self._action(side, self.BRING, obj, wait=True)

    def _update_context(self, action):
        self.context[self.actions_in_context.index(action)] = 0
