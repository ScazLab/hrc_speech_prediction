#!/usr/bin/env python

import argparse
import os
import re
from threading import Lock

import numpy as np
from sklearn.externals import joblib

import rospy
from hrc_speech_prediction import train_models as train
from hrc_speech_prediction.models import CombinedModel as cm
from human_robot_collaboration.controller import BaseController
from std_msgs.msg import String
from std_srvs.srv import Empty

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

parser.add_argument(
    '-l',
    '--learning',
    help='How to train the model',
    choices=['incremental', 'offline'],
    default='incremental')

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
        # return an object that is in context and which name is in utter
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
    ROSBAG_START = '/rosbag/start'
    ROSBAG_STOP = '/rosbag/stop'
    MIN_WORDS = 4

    def __init__(self,
                 path,
                 model='both',
                 speech_eps=0.15,
                 context_eps=0.15,
                 timer_path=None,
                 fit_type="incremental",
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
        rospy.loginfo("Training model...")
        self.path = path
        self.combined_model = self._train_or_load_model(
            speech_eps, context_eps, fit_type)
        rospy.loginfo("Model training COMPLETED")
        # utter vectorizer
        vectorizer_path = os.path.join(path, "vocabulary.pkl")
        self.vectorizer = joblib.load(vectorizer_path)
        # actions in order of context vector
        self.actions_in_context = self.combined_model.actions
        # List of successful actions taken, this is used to
        # train contextModel
        self.action_history = []
        # Subscriber to web topic to update context on repeated fail
        rospy.Subscriber(self.WEB_TOPIC, String, self._web_interface_cb)
        # Start and stop rosbag recording automatically
        # When controller starts and stops respectively.
        self.rosbag_start = rospy.ServiceProxy(self.ROSBAG_START, Empty)
        self.rosbag_stop = rospy.ServiceProxy(self.ROSBAG_STOP, Empty)
        self._debug = debug
        self._fit_type = fit_type
        self._ctxt_lock = Lock()
        self.X_dummy_cntx = np.ones(
            (len(self.actions_in_context)), dtype='bool')

    def _run(self):
        self.rosbag_start()
        self.timer.start()
        rospy.loginfo('Starting autonomous control')
        self._reset_wrong_actions()
        utter = None
        while not self.finished:
            rospy.loginfo('Waiting for utter')
            utter = self.listen_sub.wait_for_msg(timeout=20.)
            if not self._ok_baxter(utter) or len(
                    utter.split()) < self.MIN_WORDS:
                rospy.loginfo(
                    'Skipping utter (too short or not well formed): {}'.format(
                        utter))
            else:
                action, _ = self.combined_model.predict(
                    self.action_history,
                    utter=utter,
                    exclude=self.wrong_actions,
                    plot=self._debug)
                message = "Taking action {} for \"{}\"".format(action, utter)
                rospy.loginfo(message)
                self.timer.log(message)
                if self.take_action(action):
                    # Learn on successful action taken
                    if self._fit_type == "incremental":
                        self.combined_model.partial_fit([self.action_history],
                                                        utter, [action])
                    self.action_history.append(action)
                    self._reset_wrong_actions()

    def _abort(self):

        self._save_model()
        controller.rosbag_stop()  # Stops rosbag recording

        super(SpeechPredictionController, self)._abort()

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
            self.X_dummy_cntx[self.actions_in_context.index(action)] = 0
            message = 'New context: {}'.format(" ".join([
                self._short_action(self.actions_in_context[i])
                for i in self.X_dummy_cntx.nonzero()[0]
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

    @staticmethod
    def _ok_baxter(utter):
        "Checks that utter starts with something like Ok Baxter..."
        if utter:
            return re.search("^(hey|ok|okay|hi|alright|all right) baxter",
                             utter.lower())
        else:
            return False  # than utter is probably None

    def _train_or_load_model(self, speech_eps, context_eps, fit_type):
        path = os.path.join(self.path, args.participant)

        combined_model = train.train_combined_model(
            speech_eps, context_eps, fit_type=fit_type)

        if os.path.exists(path):
            n_dirs = len([d for d in os.listdir(path) if ".pkl" not in d])
            path = os.path.join(path, "update_{}".format(n_dirs - 1))

            rospy.loginfo("Loading model from {}".format(path))
            try:
                vectorizer = joblib.load(os.path.join(path, "vectorizer.pkl"))
                speech = joblib.load(os.path.join(path, "speech_model.pkl"))
                cntxt = joblib.load(os.path.join(path, "context_model.pkl"))

                combined_model._vectorizer = vectorizer
                combined_model.speech_model = speech
                combined_model.context_model = cntxt
            except (OSError, IOError):
                rospy.logerror(
                    "Error, pickled models not found! Training from data instead."
                )

        return combined_model

    def _save_model(self):
        path = os.path.join(self.path, args.participant)

        if not os.path.exists(path):
            path = os.path.join(path, "update_0")
        elif os.path.exists(path) and self._fit_type == "incremental":
            n_dirs = len([d for d in os.listdir(path) if ".pkl" not in d])
            path = os.path.join(path, "update_{}".format(n_dirs))
        else:
            return

        os.makedirs(path)

        rospy.loginfo("Saving models to {}".format(path))

        with open(os.path.join(path, "vectorizer.pkl"), "wb") as f:
            joblib.dump(self.combined_model._vectorizer, f, compress=9)
        with open(os.path.join(path, "speech_model.pkl"), "wb") as f:
            joblib.dump(self.combined_model.speech_model, f, compress=9)
        with open(os.path.join(path, "context_model.pkl"), "wb") as f:
            joblib.dump(self.combined_model.context_model, f, compress=9)


args = parser.parse_args()
controller = SpeechPredictionController(
    path=args.path,
    debug=args.debug,
    model=args.model,
    fit_type=args.learning,
    timer_path='timer-{}.json'.format(args.participant))

controller.run()
