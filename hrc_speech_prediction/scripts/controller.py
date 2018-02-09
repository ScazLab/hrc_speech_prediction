#!/usr/bin/env python

import argparse
import os
import re
from threading import Lock

import numpy as np
from sklearn.linear_model import SGDClassifier

import rospy
from hrc_speech_prediction import train_models as train
from hrc_speech_prediction.data import ALL_ACTIONS
from hrc_speech_prediction.models import CombinedModel as CM
from hrc_speech_prediction.models import JointModel
from human_robot_collaboration.controller import BaseController
from human_robot_collaboration_msgs.msg import DataLog
from std_msgs.msg import String
from std_srvs.srv import Empty

parser = argparse.ArgumentParser("Run the autonomous controller")
parser.add_argument(
    'path', help='path to the model files', default=os.path.curdir)
parser.add_argument(
    '-m',
    '--model-from-trial',
    type=int,
    default=None,
    help='path to model to use')
parser.add_argument(
    '-p', '--participant', help='id of participant', default='test')
parser.add_argument(
    '-t', '--trial', type=int, help='id of the trial', default='test')

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

TFIDF = False
N_GRAMS = (1, 2)
SPEECH_MODEL_PARAMETERS = {
    'alpha': .04,
    'loss': 'log',
    'average': True,
    'penalty': 'l2',
}


def _check_path(path, fail_on_exist=True):
    if os.path.exists(path):
        if not os.path.isdir(path):
            raise IOError('Path exists and is not a directory')
        elif fail_on_exist:
            raise IOError('Path to trial already exists')
    else:
        os.makedirs(path)


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
                 participant='test',
                 trial=0,
                 model=None,
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
        # Initializing storage space
        participant_path = os.path.join(path, participant)
        _check_path(participant_path, fail_on_exist=False)
        self.trial = trial
        self.path = os.path.join(participant_path, str(trial))
        _check_path(self.path)
        rospy.loginfo("Training model...")
        if model is None:
            self._train_model(speech_eps, context_eps, fit_type)
        else:
            model = os.path.join(participant_path, str(model), "model_final")
            self._load_model(model, speech_eps, context_eps)
        rospy.loginfo("Model training COMPLETED")
        # actions in order of context vector
        self.actions_in_context = self.model.actions
        # List of successful actions taken, this is used to
        # train contextModel
        self.action_history = []
        # Subscriber to web topic to update context on repeated fail
        rospy.Subscriber(self.WEB_TOPIC, String, self._web_interface_cb)
        # Start and stop rosbag recording automatically
        # When controller starts and stops respectively.
        self.rosbag_start = rospy.ServiceProxy(self.ROSBAG_START, Empty)
        self.rosbag_stop = rospy.ServiceProxy(self.ROSBAG_STOP, Empty)

        self.data_pub = rospy.Publisher(
            'controller_data', DataLog, queue_size=10)
        #rospy.init_node('controller', anonymous=True)

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
            self.log_msg = DataLog(
            )  # Logs the outcome of each speech/action pair
            rospy.loginfo('Waiting for utter')
            utter = self.listen_sub.wait_for_msg(timeout=20.)
            if not self._ok_baxter(utter) or len(
                    utter.split()) < self.MIN_WORDS:
                rospy.loginfo(
                    'Skipping utter (too short or not well formed): {}'.format(
                        utter))
            else:
                action, _ = self.model.predict(
                    self.action_history,
                    utter=utter,
                    exclude=self.wrong_actions,
                    plot=self._debug)
                message = "Taking action {} for \"{}\"".format(action, utter)
                rospy.loginfo(message)
                self.timer.log(message)

                self.log_msg.action = action
                self.log_msg.utter = utter
                if self.take_action(action):
                    # Learn on successful action taken
                    if self._fit_type == "incremental":
                        self.model.partial_fit([self.action_history], utter,
                                               [action])
                    self.action_history.append(action)
                    self._reset_wrong_actions()

                self.data_pub.publish(self.log_msg)

    def _abort(self):

        self._save_model('model_final')
        controller.rosbag_stop()  # Stops rosbag recording

        super(SpeechPredictionController, self)._abort()

    def take_action(self, action):
        side, obj = self.OBJECT_DICT[action]
        for _ in range(3):  # Try four time to take action
            r = self._action(side, (self.BRING, [obj]), {'wait': True})
            rospy.loginfo("TAKE ACTION ERROR: {}".format(r.response))

            if r.success:
                self.log_msg.result = self.log_msg.CORRECT
                return True

            elif r.response == r.ACT_FAILED:
                message = "Marking {} as a wrong answer (adding to: [{}])".format(
                    action, ", ".join(
                        map(self._short_action, self.wrong_actions)))

                rospy.loginfo(message)
                self.timer.log(message)

                with self._ctxt_lock:
                    self.wrong_actions.append(action)

                self.log_msg.result = self.log_msg.FAIL
                return False

            elif r.response in (r.NO_IR_SENSOR, r.ACT_NOT_IMPL):
                rospy.logerr(r.response)
                self._stop()

            else:
                # Otherwise retry action
                rospy.logwarn('Retrying failed action {}. [{}]'.format(
                    action, r.response))

        self.log_msg.result = self.log_msg.ERROR
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

    def _load_model(self, model_path, speech_eps, context_eps):
        self.model = CM.load_from_path(model_path, ALL_ACTIONS,
                                       JointModel.model_generator(
                                           SGDClassifier,
                                           **SPEECH_MODEL_PARAMETERS),
                                       speech_eps, context_eps)

    def _train_model(self, speech_eps, context_eps, fit_type):
        self.model = train.train_combined_model(
            speech_eps,
            context_eps,
            fit_type=fit_type,
            tfidf=TFIDF,
            n_grams=N_GRAMS,
            speech_model_class=SGDClassifier,
            speech_model_parameters=SPEECH_MODEL_PARAMETERS,
            init_new_speech_actions=(fit_type == "incremental"))
        self._save_model('model_initial')

    def _save_model(self, name='model'):
        self.model.save(os.path.join(self.path, name))
        rospy.loginfo("Saved models {} to {}".format(name, self.path))


args = parser.parse_args()
controller = SpeechPredictionController(
    path=args.path,
    participant=args.participant,
    trial=args.trial,
    debug=args.debug,
    model=args.model_from_trial,
    fit_type=args.learning,
    timer_path='timer-{}.json'.format(args.participant))

controller.run()
