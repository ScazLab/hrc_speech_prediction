import os
import argparse

import numpy as np
from scipy.interpolate import spline
import matplotlib.pyplot as plt
from sklearn import metrics

from hrc_speech_prediction import features
from hrc_speech_prediction.data import TrainData, TRAIN_PARTICIPANTS


parser = argparse.ArgumentParser("Train and evaluate classifier")
parser.add_argument(
    'path', help='path to the experiment data', default=os.path.curdir)


class Evaluation(object):
    """Evaluates a given model on the specified data.

    Runs a number of different evaluations (per-user, ...).
    """

    def __init__(self, model_generator, data_path, n_grams=(1, 1), tfidf=False,
                 model_variations=None):
        self.data = TrainData.load(os.path.join(data_path, "train.json"))
        self.model_gen = model_generator
        self.model_variations = model_variations
        self.X_context, self.context_actions = \
            features.get_context_features(self.data)
        self.X_speech, _ = features.get_bow_features(
            self.data, tfidf=tfidf, n_grams=n_grams)
        # Participants we will exclude from training, but will test on
        self.test_participants = ["15.ADT"]

    def get_Xs(self, indices):
        return self.X_context[indices, :], self.X_speech[indices, :]

    def get_labels(self, indices):
        return [list(self.data.labels)[i] for i in indices]

    def check_indices(self, train_idx, test_idx):
        assert (not set(train_idx).intersection(test_idx))

    def evaluate_on_one_participant(self):
        """
        Leaves one participant out of training and then tests on them. Does
        this for each participant
        """
        print("Running test on one participant...")
        participants = TRAIN_PARTICIPANTS
        results = {}
        # Get the indices for training and testing data
        for tst in participants:
            test_idx = list(self.data.data[tst].ids)
            train_idx = [i for p in participants
                         for i in list(self.data.data[p].ids)
                         if not p == tst]
            results[tst] = self._evaluate_each_model_on(train_idx, test_idx)
        self._print_result_table(results, "Participants")
        return [t for t in results.values()]

    def evaluate_on_one_trial(self):
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
                for trial in self.data.data[part]
                for i in trial.ids if tst == trial.instruction
            ]
            train_idx = [i for p in TRAIN_PARTICIPANTS
                         for i in self.data.data[p].ids if i not in test_idx]
            results[tst] = self._evaluate_each_model_on(train_idx, test_idx)
        self._print_result_table(results, "Instruction")

    def cross_validation(self):
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
            results.append(self._evaluate_each_model_on(train_idx, test_idx))
        self._print_result_table(dict(enumerate(results)), '10-fold CV')

    def new_participant(self):
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
            results[t] = self._evaluate_each_model_on(train_idx, test_idx)
        self._print_result_table(results, 'Instruction')

    def evaluate_all(self):
        self.evaluate_on_one_participant()
        self.evaluate_on_one_trial()
        self.cross_validation()
        self.new_participant()

    def evaluate_incremental_learning(self):
        classes = np.unique((self.data.labels))
        score = []
        for i in range(0, self.X_context.shape[0] - 1):
            # Train set
            train_X = self.get_Xs([i])
            train_Y = self.get_labels([i])
            # Test set
            test_X = self.get_Xs([i + 1])
            test_Y = self.get_labels([i + 1])
            model = self.model_gen(
                self.context_actions,
                randomize_context=.25
            ).partial_fit(train_X[0], train_X[1], train_Y, classes)
            prediction = model.predict(*test_X)
            score.append(metrics.accuracy_score(
                test_Y, prediction, normalize=True, sample_weight=None))

        print(score)
        xaxis = np.array([i for i in range(0, self.X_context.shape[0] - 1)])
        score = np.array(score)

        xnew = np.linspace(xaxis.min(), xaxis.max(), 1000)
        score_smooth = spline(xaxis, score, xnew)

        plt.plot(xnew, score_smooth)
        plt.show()

    @property
    def _variations(self):
        if self.model_variations is None:
            return {'Accuracy': {}}
        else:
            return self.model_variations

    def _evaluate_each_model_on(self, train_idx, test_idx):
        return {m: self._evaluate_on(train_idx, test_idx,
                                     model_params=self._variations[m])
                for m in self._variations}

    def _evaluate_on(self, train_idx, test_idx, model_params):
        self.check_indices(train_idx, test_idx)
        # Train set
        train_Xs = self.get_Xs(train_idx)
        train_Y = self.get_labels(train_idx)
        # Test set
        test_Xs = self.get_Xs(test_idx)
        test_Y = self.get_labels(test_idx)
        # Train
        model = self.model_gen(self.context_actions,
                               randomize_context=.25,
                               **model_params
                               ).fit(train_Xs[0], train_Xs[1], train_Y)
        # Evaluate
        prediction = model.predict(*test_Xs)
        return metrics.accuracy_score(
            test_Y, prediction, normalize=True, sample_weight=None)

    def _reorder_by_model(self, results):
        return {m: {k: results[k][m] for k in results} for m in self._variations}

    def _print_result_table(self, results, key_title):
        results = self._reorder_by_model(results)
        result_keys = list(results.values())[0].keys()
        txt_cell = "{:^7}"
        num_cell = "{:^7.2f}"
        avg_cell = "{:^7.3f}"
        std_cell = "{:^7.4f}"
        w = max(len(key_title), len("Accuracy"))
        print("{:<{w}}: {} ".format(
            key_title,
            " ".join([txt_cell.format(k) for k in result_keys] +
                     [txt_cell.format("AVG."), txt_cell.format("STD.")]),
            w=w))
        for m in self._variations:
            print("{:<{w}}: {} ".format(
                m,
                " ".join(
                    [num_cell.format(v) for v in results[m].values()] +
                    [avg_cell.format(np.average(results[m].values())),
                     std_cell.format(np.std(results[m].values()))]),
                w=w))
