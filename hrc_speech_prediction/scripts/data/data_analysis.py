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
    '11.ACpCp': (1),
    '12.ACpCp': (1),
    '15.ACpCp': (1, 2, 3)
}  # Incomplete trials

# 10 had to abort two pieces before

TRIALS = set((1, 2, 3))


class AnalyzeData(object):
    def __init__(self, exclude):
        self.exclude = exclude

    def _filter_bags(self):
        """Get relevant bags for parsing for each trial"""
        bags_by_trial = {1: [], 2: [], 3: []}

        for root, dirs, filenames in os.walk(args.bag_path):
            for p in dirs:
                try:
                    excluded_trials = set(self.exclude[p])
                except KeyError:
                    excluded_trials = set(())
                if p == "PILOTS":
                    continue

                bags = list(participant_bags(args.bag_path, p))

                for i in (TRIALS - excluded_trials):
                    bags_by_trial[i].append(bags[i - 1])

        return bags_by_trial

    def count_errors_across_trials(self, bag_dict):
        trial1_errors = self._bags_to_error_counts(bag_dict[1])
        trial2_errors = self._bags_to_error_counts(bag_dict[2])
        trial3_errors = self._bags_to_error_counts(bag_dict[3])
        return trial1_errors, trial2_errors, trial3_errors

    def _bags_to_error_counts(self, bags):
        error_counts = []
        for bag in bags:
            count = 0
            for m in bag.read_messages():
                if m.topic == TOPIC:
                    if m.result == "ERROR":
                        count += 1
            error_counts.append(count)
        return np.array(error_counts)

    def plot_across_trials(self):
        pass


args = parser.parse_args()

a = AnalyzeData(EXCLUDE)
print(a._filter_bags())
