#!/usr/bin/env python

import argparse
import os
from threading import Lock

import numpy as np
from sklearn.externals import joblib

import rospy
from hrc_speech_prediction.models import combined_model as cm
from human_robot_collaboration.controller import BaseController
from std_msgs.msg import String

parser = argparse.ArgumentParser("Run the autonomous controller")
parser.add_argument(
    'path', help='path to the model files', default=os.path.curdir)
parser.add_argument(
    '-m',
    '--model',
    help='model to use',
    choices=['speech', 'both', 'speech_table', 'both_table'],
    default='both')
parser.add_argument(
    '-p', '--participant', help='id of participant', default='test')

parser.add_argument(
    '-d',
    '--debug',
    help='displays plots for each predicition',
    dest='debug',
    action='store_true')
parser.set_defaults(debug=False)


class DummyPredictor(object):
    def __init__(self, object_list):
        self.obj = object_list
        self.words = [o.split('_')[0] for o in self.obj]

    @property
    def n_obj(self):
        return len(self.obj)

    def transform(self, utterances):
        return np.array(
            [[w in u.lower() for w in self.words] for u in utterances])

    def predict(self, Xc, Xs, exclude=[]):
        # return an object that is in context and which name is in utterance
        intersection = Xs * Xc
        intersection[:, [self.obj.index(a) for a in exclude]] = 0
        chosen = -np.ones((Xc.shape[0]), dtype='int8')
        ii, jj = intersection.nonzero()
        chosen[ii] = jj
        scr = self.obj.index('screwdriver_1')
        chosen[(chosen == -1).nonzero()[0]] = scr
        return [self.obj[c] for c in chosen]


class SpeechPredictionController(BaseController):

    OBJECT_DICT = {
        "seat": (BaseController.LEFT, 198),
        "chair_back": (BaseController.LEFT, 201),
        "leg_1": (BaseController.LEFT, 150),
        "leg_2": (BaseController.LEFT, 151),
        "leg_3": (BaseController.LEFT, 152),
        "leg_4": (BaseController.LEFT, 153),
        "leg_5": (BaseController.LEFT, 154),
        "leg_6": (BaseController.LEFT, 155),
        "leg_7": (BaseController.LEFT, 156),
        "foot_1": (BaseController.RIGHT, 10),
        "foot_2": (BaseController.RIGHT, 11),
        "foot_3": (BaseController.RIGHT, 12),
        "foot_4": (BaseController.RIGHT, 13),
        "front_1": (BaseController.RIGHT, 14),
        "front_2": (BaseController.RIGHT, 15),
        "top_1": (BaseController.RIGHT, 16),
        "top_2": (BaseController.RIGHT, 17),
        "back_1": (BaseController.RIGHT, 18),
        "back_2": (BaseController.RIGHT, 19),
        "screwdriver_1": (BaseController.RIGHT, 20),
        "front_3": (BaseController.RIGHT, 22),
        "front_4": (BaseController.RIGHT, 23),
    }
    BRING = 'get_pass'
    WEB_TOPIC = '/web_interface/pressed'
    MIN_WORDS = 4

    def __init__(self,
                 path,
                 model='both',
                 timer_path=None,
                 debug=False,
                 **kwargs):
        super(SpeechPredictionController, self).__init__(
            left=True,
            right=True,
            speech=False,
            listen=True,
            recovery=True,
            timer_path=os.path.join(path, timer_path),
            **kwargs)
        # if debug:
        #  self.model = DummyPredictor(list(self.OBJECT_DICT.keys()))
        #  self.vectorizer = self.model
        #  self.actions_in_context = self.model.obj
        model_path = os.path.join(path, "model_{}.pkl".format(model))
        self.model = joblib.load(model_path)
        # utterance vectorizer
        vectorizer_path = os.path.join(path, "vocabulary.pkl")
        combined_model_path = os.path.join(path, "combined_model_0.150.15.pkl")
        self.vectorizer = joblib.load(vectorizer_path)
        self.combined_model = joblib.load(combined_model_path)
        # actions in order of context vector
        self.actions_in_context = self.model.actions
        # Subscriber to web topic to update context on repeated fail
        rospy.Subscriber(self.WEB_TOPIC, String, self._web_interface_cb)
        self._debug = debug
        self._ctxt_lock = Lock()
        self.context = np.ones((len(self.actions_in_context)), dtype='bool')

    def _run(self):
        self.timer.start()
        rospy.loginfo('Starting autonomous control')
        self._reset_wrong_actions()
        utterance = None
        while not self.finished:
            rospy.loginfo('Waiting for utterance')
            utterance = self.listen_sub.wait_for_msg(timeout=20.)
            if utterance is None or len(utterance.split()) < self.MIN_WORDS:
                rospy.loginfo(
                    'Skipping utterance (too short): {}'.format(utterance))
            else:
                #x_u = self.vectorizer.transform([utterance])
                action, _ = self.combined_model.take_action(
                    utter=utterance, plot=self._debug)
                # with self._ctxt_lock:
                #     ctxt = self.context.copy()
                #     action = self.model.predict(ctxt[None, :], x_u,
                #                                 exclude=self.wrong_actions)[0]
                message = "Taking action {} for \"{}\"".format(
                    action, utterance)
                rospy.loginfo(message)
                self.timer.log(message)
                if self.take_action(action):
                    self.combined_model.curr = action

                    #self._update_context(action)

    def take_action(self, action):
        side, obj = self.OBJECT_DICT[action]
        for _ in range(3):  # Try four time to take action
            r = self._action(side, (self.BRING, [obj]), {'wait': True})
            if r.success:
                return True
            elif r.response == r.ACT_FAILED:
                message = "Marking {} as a wrong answer (adding to: [{}])".format(
                    action, ", ".join(
                        map(self._short_action, self.wrong_actions)))
                rospy.loginfo(message)
                self.timer.log(message)
                with self._ctxt_lock:
                    self.wrong_actions.append(action)
                return False
            elif r.response in (r.NO_IR_SENSOR, r.ACT_NOT_IMPL):
                rospy.logerr(r.response)
                self._stop()
            else:
                # Otherwise retry action
                rospy.logwarn('Retrying failed action {}. [{}]'.format(
                    action, r.response))
        return False

    def _reset_wrong_actions(self):
        with self._ctxt_lock:
            self.wrong_actions = []

    def _web_interface_cb(self, message):
        if message.data in self.actions_in_context:
            self._update_context(message.data)
        elif message.data == 'START experiment':
            rospy.logwarn('Starting!')
            self.timer.log('Start')
        elif message.data == 'STOP experiment':
            self._stop()

    def _update_context(self, action):
        with self._ctxt_lock:
            self.context[self.actions_in_context.index(action)] = 0
            message = 'New context: {}'.format(" ".join([
                self._short_action(self.actions_in_context[i])
                for i in self.context.nonzero()[0]
            ]))
        self.timer.log(message)
        rospy.loginfo(message)
        self._reset_wrong_actions()

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
    debug=args.debug,
    model=args.model,
    timer_path='timer-{}.json'.format(args.participant))

controller.run()
