import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
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

TOPIC = "/controller_data"
EXCLUDE = {  # Number in tuples represent trials to ignore
    '4.ACpCp': (1, 2, 3),
    '5.BApAp': (1, 2, 3),
    '6.BApAp': (1, 2, 3),
    '11.ACpCp': (1, 2, 3),
    '12.ACpCp': (1, ),
    '15.ACpCp': (1, 2, 3)
}  # Incomplete trials

# 10 had to abort two pieces before

TRIALS = set((1, 2, 3))


class AnalyzeData(object):
    def __init__(self, exclude):
        self.exclude = exclude

    def _filter_bags_by_trial(self):
        """Get relevant bags for parsing for each trial"""
        bags_by_trial = {1: [], 2: [], 3: []}

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
                    bags_by_trial[i].append(bags[i - 1])
            break  # Dont want to loop through ABORTED or PILOT dirs

        return bags_by_trial

    def _count_errors_by_instruction(self):
        """Get error counts across each instruction for each trial"""
        bags_by_trial = self._filter_bags_by_trial()
        counts_by_instr = {
            1: np.zeros((20, len(bags_by_trial[1]))),
            2: np.zeros((20, len(bags_by_trial[2]))),
            3: np.zeros((20, len(bags_by_trial[3])))
        }

        for k in bags_by_trial.keys():
            j = 0
            for b in bags_by_trial[k]:
                i = 0
                for m in b.read_messages():
                    if m.topic == TOPIC:
                        if m.message.result == DataLog.CORRECT:
                            i += 1
                        if m.message.result == DataLog.FAIL:
                            counts_by_instr[k][i][j] += 1
                j += 1

        return counts_by_instr

    def _count_errors_across_trials(self):
        bag_dict = self._filter_bags_by_trial()
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
        return np.array(error_counts)

    def plot_across_trials(self):
        t1, t2, t3 = self._count_errors_across_trials()
        print(len(t1), len(t2))

        plt.figure()
        plt.title("Errors per trial")
        plt.boxplot([t1, t2, t3])

        plt.ylim(0, 8)
        plt.tight_layout()
        plt.show()

    def plot_across_instructions(self):
        counts_by_instr = self._count_errors_by_instruction()

        fig, axarr = plt.subplots(3, sharex=True, sharey=True)

        plt.suptitle("Errors per instruction", fontsize=20)

        for i in range(3):
            bp = axarr[i].boxplot(np.transpose(counts_by_instr[i + 1]))
            axarr[i].set_title("Trial {}".format(i + 1))

        plt.show()


args = parser.parse_args()

a = AnalyzeData(EXCLUDE)
a.plot_across_instructions()
