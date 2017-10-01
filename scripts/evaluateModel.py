import os
import argparse

import numpy as np
from sklearn import metrics
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

from hrc_speech_prediction import data, features


parser = argparse.ArgumentParser("Train and evaluate classifier")
parser.add_argument('path', help='path to the experiment data',
                    default=os.path.curdir)


class evaluateModel(object):

    def __init__(self, model, data_path, n_grams=(1, 1), **kwargs):
        """
        Given a model and a path to the data, will run a number of different
        evaluations
        """
        self.data = data.TrainData.load(os.path.join(data_path, "train.json"))
        self.model = model
        self.args = kwargs
        self.m_features, _ = features.get_context_features(self.data)
        self.m_speech, _ = features.get_bow_features(self.data,
                                                         tfidf=False,
                                                         n_grams=n_grams)
        self.m_all = np.concatenate(
            (self.m_features, self.m_speech.toarray()), axis=1)

    def test_on_one_participant(self, data_type="context"):
        """
        Leaves on participant out of training and then tests on them. Does
        this for each participant
        """

        print("Running test on one participant...")

        participants = self.data.participants
        results = {}
        score_avg = 0

        if data_type is "context":
            m_data = self.m_features
        elif data_type is "speech":
            m_data = self.m_speech.toarray()
        else:
            m_data = self.m_all

        # Get the indices for training and testing data
        for tst in participants:

            # Get the labels for the testing participant
            test_Y = list(self.data.data[tst].labels)

            test_idx = list(self.data.data[tst].ids)
            test_X = m_data[test_idx, :]

            train_idx = [
                list(self.data.data[p].ids) for p in participants
                if not p == tst
            ]
            train_X = m_data[[i for pi in train_idx for i in pi], :]
            train_Y = [
                list(self.data.labels)[i] for pi in train_idx for i in pi
            ]

            model = self.model().fit(train_X, train_Y)
            prediction = model.predict(test_X)

            #.reshape(1, -1
            score = metrics.accuracy_score(
                test_Y, prediction, normalize=True, sample_weight=None)
            score_avg += score

            results[tst] = score

        score_avg = score_avg / self.data.n_participants

        print("{:<12}: {} ".format("Participant", " ".join(
            ["{:^7}".format(k) for k in results.keys()])))
        print("{:<12}: {} ".format("Accuracy", " ".join(
            ["{:^7.2f}".format(v) for v in results.values()])))
        print("Average {}\n".format(score_avg))

    def test_on_one_trial(self, data_type="context"):
        """
        Excludes on trial from training (i.e. A, B, or C) and uses these
        excluded trials as tests
        """
        print("Running test on one trial...")

        trials = ['A', 'B', 'C']
        participants = self.data.participants
        score_avg = 0

        results = {}

        if data_type is "context":
            m_data = self.m_features
        elif data_type is "speech":
            m_data = self.m_speech.toarray()
        else:
            m_data = self.m_all

        for t in trials:
            # train_trials = [i for i in trials if i != t]

            test_idx = [
                i
                for part in self.data.data for trial in self.data.data[part]
                for i in trial.ids if t == trial.instruction
            ]

            train_idx = [
                i for i in range(0, self.data.n_samples) if i not in test_idx
            ]

            test_X = m_data[test_idx, :]
            train_X = m_data[train_idx, :]

            train_Y = [list(self.data.labels)[i] for i in train_idx]
            test_Y = [list(self.data.labels)[i] for i in test_idx]

            model = self.model(**self.args).fit(train_X, train_Y)
            prediction = model.predict(test_X)

            score = metrics.accuracy_score(
                test_Y, prediction, normalize=True, sample_weight=None)

            score_avg += score
            results[t] = score


        print("{:<12}: {} ".format("Participant", " ".join(
            ["{:^7}".format(k) for k in results.keys()])))
        print("{:<12}: {} ".format("Accuracy", " ".join(
            ["{:^7.2f}".format(v) for v in results.values()])))

        score_avg = score_avg / 3.0
        print("Average {0:.2f}\n".format(score_avg))

    def cross_validation(self, data_type="context"):
        """
        10-fold cross validation
        """
        print("Running 10-fold cross validation...")

        results = np.empty((1, ))

        if data_type is "context":
            m_data = self.m_features
        elif data_type is "speech":
            m_data = self.m_speech.toarray()
        else:
            m_data = self.m_all

        step_size = self.data.n_samples / 10
        for i in range(0, self.data.n_samples, step_size):

            if i + step_size <= self.data.n_samples:
                next_i = i + step_size
            else:
                next_i = self.data.n_samples

            test_idx = [j for j in range(i, next_i)]
            train_idx = [
                j for j in range(0, self.data.n_samples) if j not in test_idx
            ]

            train_X = m_data[train_idx, :]
            test_X = m_data[test_idx, :]

            train_Y = [list(self.data.labels)[j] for j in train_idx]
            test_Y = [list(self.data.labels)[j] for j in test_idx]

            model = self.model(**self.args).fit(train_X, train_Y)
            prediction = model.predict(test_X)

            score = metrics.accuracy_score(
                test_Y, prediction, normalize=True, sample_weight=None)

            results = np.append(results, score)

        print("Avg: {:.2f}, std dev: {:.2f}\n".format(
            np.mean(results), np.std(results)))

    def test_all(self):
        for data_type in ["context", "speech", "both"]:
            print("---------------testing on {}---------------"
                  .format(data_type))
            self.test_on_one_participant(data_type)
            self.test_on_one_trial(data_type)
            self.cross_validation(data_type)


if __name__ == '__main__':
    args = parser.parse_args()

    ev = evaluateModel(LogisticRegression, args.path, n_grams=(2,2))
    ev.test_all()
    # ev.test_on_one_participant()
