import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from sklearn.linear_model import SGDClassifier

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
                        if p not in bags_dict.keys():
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
        # plt.ylim(-.8, 2)

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
            axarr[i].boxplot(np.transpose(counts_by_instr[i + 1]))
            axarr[i].set_title("Trial {}".format(i + 1))

        plt.savefig(
            os.path.join(SAVE_PATH, "boxes_per_instruction_one_plot.pdf"))
        plt.show()

    def _get_model(self, participant, trial):
        if trial == 0:
            model_type = "model_initial"
            t = 0
        else:
            model_type = "model_final"
            t = trial - 1
        model_path = os.path.join(args.model_path, participant,
                                  str(t), model_type)
        return CombinedModel.load_from_path(
            model_path, ALL_ACTIONS,
            JointModel.model_generator(SGDClassifier,
                                       **SPEECH_MODEL_PARAMETERS),
            SPEECH_EPS, CONTEXT_EPS)

    def print_model_performances(self):
        bag_dict = self._filter_bags()
        participant_bags = bag_dict.keys()

        print("Format: both, speech, context; score is number of correct")
        for p, bags in participant_bags.iteritems():
            for trial, bag in enumerate(bags):
                model = self._get_model(p, trial)

                cntxt = []

                i = 0
                both_score = 0
                speech_score = 0
                context_score = 0

                last_pred_speech = None
                last_pred_context = None

                was_success = True

                for m in bag.read_messages():
                    if m.topic == TOPIC:
                        if was_success:
                            # New prediction result
                            was_success = False
                            if m.message.result == DataLog.CORRECT:
                                both_score += 1
                            last_pred_speech = model.predict(
                                cntxt, m.message.utter, model='speech',
                                plot=False)
                            last_pred_context = model.predict(
                                cntxt, m.message.utter, model='context',
                                plot=False)

                        if ((not was_success) and
                                m.message.result == DataLog.CORRECT):
                            # Resolve last speech and context prediction
                            # Might happen in the same run as the previous case
                            ground_truth = m.message.action
                            speech_score += (ground_truth == last_pred_speech)
                            context_score += (ground_truth == last_pred_context)
                            last_pred_speech, last_pred_context = None, None
                            was_success = True
                            cntxt.append(ground_truth)
                            i += 1

                if (last_pred_speech is not None or
                        last_pred_context is not None):
                    raise RuntimeError("Last predictino not resolved")

            print("{} / {}: {} {} {} / {}".format(
                p, trial, both_score, speech_score, context_score, i))


args = parser.parse_args()

with plt.rc_context(rc=PLOT_PARAMS):
    a = AnalyzeData(EXCLUDE)
    # a.plot_errs_across_instructions()
    print(len(a._filter_bags(filter_by="participant").keys()))
    # a.plot_across_trials()
    a.print_model_performances()
