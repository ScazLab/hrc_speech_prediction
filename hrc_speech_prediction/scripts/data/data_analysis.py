import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from sklearn.externals import joblib
from sklearn.linear_model import SGDClassifier

import rosbag
import rospy
from hrc_speech_prediction.bag_parsing import participant_bags
from hrc_speech_prediction.data import ALL_ACTIONS
from hrc_speech_prediction.defaults import MODEL_PATH
from hrc_speech_prediction.models import CombinedModel, JointModel
from hrc_speech_prediction_msgs.msg import DataLog

parser = argparse.ArgumentParser(
    "Displays the bag as a log of relevant information")
parser.add_argument(
    '--bag_path',
    help='path to the experiment files',
    default="/home/scazlab/Desktop/speech_prediction_bags/Experiment2Data/")
parser.add_argument(
    '--model_path', help='path to model files', default=MODEL_PATH)

TOPIC = "/controller_data"
EXCLUDE = {  # Number in tuples represent trials to ignore
    '4.ACpCp': (1, 2, 3),
    '5.BApAp': (1, 2, 3),
    '6.BApAp': (1, 2, 3),
    '11.ACpCp': (1, 2, 3),
    '12.ACpCp': (1, 2, 3),
    '15.ACpCp': (1, 2, 3)
}  # Incomplete trials

# 10 had to abort two pieces before

TRIALS = set((1, 2, 3))

RED = '#d9262c'
BLUE = '#308bc9'
GREEN = '#089164'

COLORS = [RED, BLUE, GREEN]
TRIAL_NAMES = [
    r'Familiar orderings', r'Novel orderings', r'Repeated novel orderings'
]

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

SAVE_PATH = os.path.join(os.path.dirname(__file__), "figs/results/")


class AnalyzeData(object):
    def __init__(self, exclude):
        self.exclude = exclude

    def _filter_bags(self, filter_by="trial"):
        """Get relevant bags for parsing for each trial"""
        if filter_by == "trial":
            bags_dict = {1: [], 2: [], 3: []}
        elif filter_by == "participant":
            bags_dict = {}

        for root, dirs, filenames in os.walk(args.bag_path):
            for p in dirs:
                if p in ["ABORTED", "PILOTS"]:
                    continue
                try:
                    excluded_trials = set(self.exclude[p])
                except KeyError:
                    excluded_trials = set(())

                bags = list(participant_bags(args.bag_path, p))

                for i in (TRIALS - excluded_trials):
                    if filter_by == "trial":
                        bags_dict[i].append(bags[i - 1])
                    elif filter_by == "participant":
                        if not p in bags_dict.keys():
                            bags_dict[p] = []

                        bags_dict[p].append(bags[i - 1])

            break  # Dont want to loop through ABORTED or PILOT dirs

        return bags_dict

    def _count_errors_by_instruction(self):
        """Get error counts across each instruction for each trial"""
        bags_by_trial = self._filter_bags()
        counts_by_instr = {
            1: np.zeros((20, len(bags_by_trial[1])), dtype=float),
            2: np.zeros((20, len(bags_by_trial[2])), dtype=float),
            3: np.zeros((20, len(bags_by_trial[3])), dtype=float)
        }

        for k in bags_by_trial.keys():
            for p, b in enumerate(bags_by_trial[k]):
                i = 0
                for m in b.read_messages():
                    if m.topic == TOPIC:
                        if m.message.result == DataLog.CORRECT:
                            i += 1
                        if m.message.result == DataLog.FAIL:
                            counts_by_instr[k][i][p] += 1

        return counts_by_instr

    def _count_errors_across_trials(self):
        bag_dict = self._filter_bags()
        trial1_errors = self._count_errors_by_trial(bag_dict[1])
        trial2_errors = self._count_errors_by_trial(bag_dict[2])
        trial3_errors = self._count_errors_by_trial(bag_dict[3])
        return trial1_errors, trial2_errors, trial3_errors

    def _count_errors_by_trial(self, bags):
        error_counts = []
        for bag in bags:
            count = 0
            for m in bag.read_messages():
                if m.topic == TOPIC:
                    if m.message.result == DataLog.FAIL:
                        count += 1
            error_counts.append(count)
        return np.array(error_counts, dtype=float)

    def _simplify_plot(self, ax):
        ax.spines['top'].set_visible(False)
        ax.tick_params(top=False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(right=False)

    def plot_across_trials(self):
        t1, t2, t3 = self._count_errors_across_trials()
        print("Test btw Trials 2 and 3", stats.ttest_rel(t2, t3))

        plt.figure()
        self._simplify_plot(plt.gca())

        plt.xlabel(r"Trials", fontweight='bold')
        plt.ylabel(r"Errors", fontweight='bold')

        bplot = plt.boxplot([t1, t2, t3], patch_artist=True)

        for patch, color in zip(bplot['boxes'], COLORS):
            patch.set_facecolor(color)

        plt.setp(bplot['medians'], color="red", linewidth=3.5)
        plt.setp(bplot['whiskers'], color="black")

        plt.ylim(0, 10)
        plt.grid()
        plt.tight_layout()

        plt.savefig(os.path.join(SAVE_PATH, "errors_per_trial.pdf"))
        plt.show()

    def plot_errs_across_instructions(self):
        counts_by_instr = self._count_errors_by_instruction()

        plt.figure()
        # plt.rc('text', usetex=True)
        # plt.rc('font', family='serif')

        plt.xlabel(r'Instruction steps', fontsize=20, fontweight='bold')
        plt.ylabel(r'Mean errors', fontsize=20, fontweight='bold')

        plt.xlim(.6, 21.3)
        #plt.ylim(-.8, 2)

        x = np.arange(1, 22, 1.05)
        self._simplify_plot(plt.gca())

        for i, c in zip([1, 2, 3], COLORS):
            m = np.mean(counts_by_instr[i], axis=1)
            err = np.std(counts_by_instr[i], axis=1)
            err = np.vstack((np.array(
                [i - e if i - e > 0. else i for i, e in zip(m, err)]), err))

            plt.errorbar(
                x,
                m,
                yerr=err,
                fmt='--o',
                label="Trial {}".format(i),
                ls='solid',
                lw=3,
                color=c,
                elinewidth=1,
                capthick=1)

        plt.grid()
        plt.legend(TRIAL_NAMES, loc=2)

        plt.tight_layout()
        plt.savefig(
            os.path.join(SAVE_PATH, "errors_per_instruction_one_plot.pdf"))
        plt.show()

    def plot_boxes_across_instructions(self):

        counts_by_instr = self._count_errors_by_instruction()
        fig, axarr = plt.subplots(3, sharex=True, sharey=True)

        plt.suptitle("Errors per instruction", fontsize=20)

        for i in range(3):
            bp = axarr[i].boxplot(np.transpose(counts_by_instr[i + 1]))
            axarr[i].set_title("Trial {}".format(i + 1))

        plt.savefig(
            os.path.join(SAVE_PATH, "boxes_per_instruction_one_plot.pdf"))
        plt.show()

    def plot_model_performances(self):
        bag_dict = self._filter_bags()
        participant_bags = bag_dict.keys()

        for p, bags in participant_bags.iteritems():
            for trial, bag in enumerate(bags)
                if trial == 0:
                    model_type = "model_initial"
                else:
                    model_type = "model_final"

                model_path = os.path.join(args.model_path, p,
                                        str(trial + 1), model_type)

                model = CombinedModel.load_from_path(
                    model_path, ALL_ACTIONS,
                    JointModel.model_generator(SGDClassifier,
                                            **SPEECH_MODEL_PARAMETERS),
                    SPEECH_EPS, CONTEXT_EPS)

                cntxt = []

                both_score = 0
                speech_score = 0 
                context_score = 0

                success_tracker = True

                for m in bag.read_messages():
                    if m.topic == TOPIC:
                        utter = m.message.utter
                        if m.message.result == DataLog.CORRECT:
                            both_score += 1
                        model.predict(cntxt, m.message.utter, plot=True)

                        cntxt.append(m.message.action)
                        i += 1

                        plt.tight_layout()

                        path = os.path.join(fig_path, "sample_{}_{}".format(
                            m.message.result, i))
                        plt.savefig(path, format="pdf")
                        plt.clf()


args = parser.parse_args()

with plt.rc_context(rc=PLOT_PARAMS):
    a = AnalyzeData(EXCLUDE)
    # a.plot_errs_across_instructions()
    print(len(a._filter_bags(filter_by="participant").keys()))
    #a.plot_across_trials()
