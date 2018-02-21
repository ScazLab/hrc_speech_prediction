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

EXCLUDE = {  # Number in tuples represent trials to ignore
    '4.ACpCp': (1, 2, 3),
    '5.BApAp': (1, 2, 3),
    '6.BApAp': (1, 2, 3),
    '11.ACpCp': (1, 2, 3),
    '12.ACpCp': (1),
    '15.ACpCp': (1, 2, 3)
}  # Incomplete trials

# 10 had to abort two pieces before


class AnalyzeData(object):
    def __init__(self, exclude):
        self.exclude = exclude

    def _filter_bags(self):
        """Get relevant bags for parsing for each trial"""
        for root, dirs, filenames in os.walk(args.bag_path):
            for p in dirs:
                if p == "PILOTS":
                    continue
                bags = participant_bags(args.bag_path, p)
                filt_bags = [bags[i - 1 for i in self.exclude]]

    def plot_across_trials(self):
        pass


args = parser.parse_args()
