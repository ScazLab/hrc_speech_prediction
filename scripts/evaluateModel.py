import os
import argparse

import numpy as np
import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier

from hrc_speech_prediction import features
from hrc_speech_prediction.data import (TrainData,
                                        TRAIN_PARTICIPANTS,
                                        ALL_ACTIONS)
from hrc_speech_prediction.models import (BaseModel,
                                          ContextFilterModel,
                                          PragmaticModel)


parser = argparse.ArgumentParser("Train and evaluate classifier")
parser.add_argument(
    'path', help='path to the experiment data', default=os.path.curdir)


class EvaluateModel(object):
    def __init__(self,
                 model,
                 data_path,
                 lam=1,
                 n_grams=(1, 1),
                 tfidf=False,
                 online=False):
        """
        Given a model and a path to the data, will run a number of different
        evaluations
        """
        self.data = TrainData.load(os.path.join(data_path, "train.json"))
        self.model = model
        self.online = online
        self.X_context, self.context_actions = \
            features.get_context_features(self.data)
        self.X_speech, _ = features.get_bow_features(
            self.data, tfidf=tfidf, n_grams=n_grams)
        # Participants we will exclude from testing, but will train on
        self.test_participants = ["15.ADT"]
        self.lam = lam

    def get_Xs(self, indices):
        return self.X_context[indices, :], self.X_speech[indices, :]

    def get_labels(self, indices):
        return [list(self.data.labels)[i] for i in indices]

    def check_indices(self, train_idx, test_idx):
        assert (not set(train_idx).intersection(test_idx))

    def test_on_one_participant(self, data_type="context"):
        """
        Leaves on participant out of training and then tests on them. Does
        this for each participant
        """
        print("Running test on one participant...")
        participants = TRAIN_PARTICIPANTS
        results = {}
        # Get the indices for training and testing data
        for tst in participants:
            test_idx = list(self.data.data[tst].ids)
            train_idx = [
                i for p in participants for i in list(self.data.data[p].ids)
                if not p == tst
            ]
            results[tst] = self._evaluate_on(train_idx, test_idx, data_type)
        self._print_result_table(results, "Participants")

    def test_on_one_trial(self, data_type="context"):
        """
        Excludes on trial from training (i.e. A, B, or C) and uses these
        excluded trials as tests
        """
        print("Running test on one trial...")
        results = {}
        for tst in ['A', 'B', 'C']:
            test_idx = [
                i
                for part in TRAIN_PARTICIPANTS
                for trial in self.data.data[part] for i in trial.ids
                if tst == trial.instruction
            ]
            train_idx = [
                i for p in TRAIN_PARTICIPANTS for i in self.data.data[p].ids
                if i not in test_idx
            ]
            results[tst] = self._evaluate_on(train_idx, test_idx, data_type)
        self._print_result_table(results, "Instruction")

    def cross_validation(self, data_type="context"):
        """
        10-fold cross validation
        """
        print("Running 10-fold cross validation...")
        n_samples = sum(
            [len(list(self.data.data[p].ids)) for p in TRAIN_PARTICIPANTS])
        results = []
        step_size = self.data.n_samples / 10
        for i in range(0, n_samples, step_size):
            next_i = min(i + step_size, n_samples)
            test_idx = [j for j in range(i, next_i)]
            train_idx = [j for j in range(0, n_samples) if j not in test_idx]
            results.append(self._evaluate_on(train_idx, test_idx, data_type))
        self._print_global_results(results)

    def new_participant(self, data_type="context"):
        print("Testing on pilot participant...")
        results = {}
        trials = ["A", "D", "T"]
        for t in trials:
            train_idx = [
                i for p in TRAIN_PARTICIPANTS for i in self.data.data[p].ids
            ]
            test_idx = [
                i
                for part in self.test_participants
                for trial in self.data.data[part] for i in trial.ids
                if t == trial.instruction
            ]
            results[t] = self._evaluate_on(train_idx, test_idx, data_type)
        self._print_result_table(results, 'Instruction')

    def test_all(self):
        for data_type in ["context", "speech", "both"]:
            print("---------------testing on {}---------------"
                  .format(data_type))
            self.test_on_one_participant(data_type)
            self.test_on_one_trial(data_type)
            self.cross_validation(data_type)
            self.new_participant(data_type)

    def test_incremental_learning(self, data_type="context"):
        classes = np.unique((self.data.labels))
        score = []
        for i in range(0, self.X_context.shape[0] - 1):
            # Train set
            train_X = self.get_Xs([i])
            train_Y = self.get_labels([i])
            # Test set
            test_X = self.get_Xs([i + 1])
            test_Y = self.get_labels([i + 1])
            model = self.model(self.context_actions,
                               features=data_type,
                               randomize_context=.25).partial_fit(train_X[0],
                                                                  train_X[1],
                                                                  train_Y,
                                                                  classes)
            prediction = model.predict(*test_X)
            score.append(metrics.accuracy_score(
                test_Y, prediction, normalize=True, sample_weight=None))

        xaxis = np.array([i for i in range(0, self.X_context.shape[0])])
        score = np.toarray(score)
        plt.plot(xaxis, score)
        plt.show()

    def _evaluate_on(self, train_idx, test_idx, data_type):
        self.check_indices(train_idx, test_idx)
        # Train set
        train_Xs = self.get_Xs(train_idx)
        train_Y = self.get_labels(train_idx)
        # Test set
        test_Xs = self.get_Xs(test_idx)
        test_Y = self.get_labels(test_idx)
        # Train
        model = self.model(
            self.context_actions,
            features=data_type,
            randomize_context=.25, ).fit(train_Xs[0], train_Xs[1],
                                         train_Y, online=self.online)
        # Evaluate
        prediction = model.predict(*test_Xs)
        return metrics.accuracy_score(
            test_Y, prediction, normalize=True, sample_weight=None)

    def _print_result_table(self, results, key_title):
        w = max(len(key_title), len("Accuracy"))
        print("{:<{w}}: {} ".format(
            key_title,
            " ".join(["{:^7}".format(k) for k in results.keys()]),
            w=w))
        print("{:<{w}}: {} ".format(
            "Accuracy",
            " ".join(["{:^7.2f}".format(v) for v in results.values()]),
            w=w))
        self._print_global_results(results.values())

    def _print_global_results(self, results):
        print("Average {:.3f}, std: {:.4f}\n".format(
            np.average(results), np.std(results)))


if __name__ == '__main__':
    args = parser.parse_args()

    ev = EvaluateModel(
        ContextFilterModel.model_generator(SGDClassifier,
                                           loss='log',
                                           average=True,
                                           penalty='l1'),
        args.path,
        lam=0.5,
        n_grams=(1, 2),
        online=False
    )
    ev.test_incremental_learning("both")
    # ev.test_on_one_participant()
