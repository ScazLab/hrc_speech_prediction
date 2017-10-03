#!/usr/bin/env python

import os
import argparse

import rospy
import numpy as np
from sklearn.externals import joblib

from human_robot_collaboration.controller import BaseController

parser = argparse.ArgumentParser("Run the autonomous controller")
parser.add_argument('path', help='path to the model files', default=os.path.curdir)
parser.add_argument('-m', '--model', help='model to use', choices=['speech', 'both'],
                    default='both')


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

    def predict(self, Xc, Xs, exclude=[]):
        # return an object that is in context and which name is in utterance
        intersection = Xs * Xc
        intersection[:, [self.obj.index(a) for a in exclude]] = 0
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

    def __init__(self, path, model='both', timer_path=None, debug=False):
        super(SpeechPredictionController, self).__init__(
            left=True, right=True, speech=False, listen=True, recovery=True)
        if debug:
            self.model = DummyPredictor(list(self.OBJECT_DICT.keys()))
            self.vectorizer = self.model
            self.actions_in_context = self.model.obj
        else:
            model_path = os.path.join(path, "model_{}.pkl".format(model))
            self.model = joblib.load(model_path)
            # utterance vectorizer
            vectorizer_path = os.path.join(path, "vocabulary.pkl")
            self.vectorizer = joblib.load(vectorizer_path)
            # actions in order of context vector
            self.actions_in_context = self.model.actions

    def _run(self):
        rospy.loginfo('Starting autonomous control')
        self.context = np.ones((len(self.actions_in_context)), dtype='bool')
        self.wrong_actions = []
        while not self.finished:
            utterance = None
            while not utterance:
                rospy.loginfo('Waiting for utterance')
                utterance = self.listen_sub.wait_for_msg(timeout=60.)
                if utterance is None or len(utterance) < 5:
                    rospy.loginfo('Skipping utterance (too short): {}'.format(utterance))
            x_u = self.vectorizer.transform([utterance])
            action = self.model.predict(self.context[None, :], x_u,
                                        exclude=self.wrong_actions)[0]
            rospy.loginfo("Taking action {} for \"{}\"".format(action, utterance))
            rospy.logwarn("Service returned {}".format(self.take_action(action)))
            self._update_context(action)

    def take_action(self, action):
        side, obj = self.OBJECT_DICT[action]
        for _ in range(3):  # Try four time to take action
            r = self._action(side, (self.BRING, [obj]), {'wait': True})
            if r.success:
                break
            elif r.response == r.ACT_FAILED:
                rospy.loginfo("Marking {} as a wrong answer (adding to: {})".format(
                    action, map(self._short_action, self.wrong_actions)))
                self.wrong_actions.append(action)
                break
            elif r.response in (r.NO_IR_SENSOR, r.ACT_NOT_IMPL):
                rospy.logerr(r.response)
                self._stop()
            else:
                # Otherwise retry action
                rospy.logwarn('Retrying failed action {}. [{}]'.format(
                    action, r.response))
        # IMPORTANT: Assuming action success after three failures
        self.wrong_actions = []


    def _update_context(self, action):
        self.context[self.actions_in_context.index(action)] = 0
        rospy.loginfo('New context: {}'.format(" ".join([
            self._short_action(self.actions_in_context[i])
            for i in self.context.nonzero()[0]])))

    @staticmethod
    def _short_action(a):
        # Printing 2 first and last char
        if a == 'seat':
            return a
        else:
            return ''.join([a[:2], a[-1]])


args = parser.parse_args()
controller = SpeechPredictionController(
    path=args.path,
    model=args.model,
    timer_path=os.path.join(args.path, 'timer.json'))

controller.run()
