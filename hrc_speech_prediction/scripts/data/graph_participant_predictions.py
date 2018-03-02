import argparse
import os

import matplotlib.pyplot as plt
from sklearn.externals import joblib
from sklearn.linear_model import SGDClassifier

import rosbag
import rospy
from hrc_speech_prediction.bag_parsing import participant_bags
from hrc_speech_prediction.data import ALL_ACTIONS
from hrc_speech_prediction.defaults import MODEL_PATH
from hrc_speech_prediction.models import CombinedModel, JointModel

parser = argparse.ArgumentParser(
    "Displays the bag as a log of relevant information")
parser.add_argument(
    '--bag_path', help='path to the experiment files', default=os.path.curdir)
parser.add_argument(
    '--model_path', help='path to model files', default=MODEL_PATH)
parser.add_argument(
    '-p', '--participant', help='id of participant', default='test')
parser.add_argument(
    '-t', '--trial', type=int, help='id of the trial', default='test')

TOPIC = "/controller_data"
TFIDF = False
N_GRAMS = (1, 2)
SPEECH_EPS = 0.15
CONTEXT_EPS = 0.15
SPEECH_MODEL_PARAMETERS = {
    'alpha': .04,
    'loss': 'log',
    'average': True,
    'penalty': 'l2',
}

PLOT_PARAMS = {
    'font.family': 'serif',
    'font.size': 10.0,
    'font.serif': 'Computer Modern Roman',
    'text.usetex': 'True',
    'text.latex.unicode': 'True',
    'axes.titlesize': 'large',
    'axes.labelsize': 'large',
    'legend.fontsize': 'medium',
    'xtick.labelsize': 'small',
    'ytick.labelsize': 'small',
    'path.simplify': 'True',
    'savefig.bbox': 'tight',
    'figure.figsize': (7.5, 4),
}


def plot_trial(trial, bag):
    if trial == 1:
        model_type = "model_initial"
    else:
        model_type = "model_final"

    model_path = os.path.join(args.model_path, args.participant, str(trial),
                              model_type)
    fig_path = os.path.join(
        os.path.dirname(__file__), "figs", args.participant, str(trial))

    if not os.path.exists(fig_path):
        os.makedirs(fig_path)

    model = CombinedModel.load_from_path(model_path, ALL_ACTIONS,
                                         JointModel.model_generator(
                                             SGDClassifier,
                                             **SPEECH_MODEL_PARAMETERS),
                                         SPEECH_EPS, CONTEXT_EPS)

    row = 4
    col = 6

    cntxt = []
    i = 0

    for m in bag.read_messages():
        if m.topic == TOPIC:
            model.predict(cntxt, m.message.utter, plot=True)

            cntxt.append(m.message.action)
            i += 1

            plt.tight_layout()

            path = os.path.join(fig_path, "sample_{}_{}".format(
                m.message.result, i))
            plt.savefig(path, format="pdf")
            plt.clf()


def plot_all_trials(bags):
    for i, b in enumerate(bags):
        plot_trial(i + 1, b)


args = parser.parse_args()

data_path = args.bag_path
BAGS = list(participant_bags(data_path, args.participant))
trial = args.trial

plot_all_trials(BAGS)
#plot_trial(trial, BAGS[trial])
