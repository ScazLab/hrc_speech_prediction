import os
import argparse

import numpy as np
import matplotlib.pyplot as plt

import random

from scipy.interpolate import spline

from sklearn import metrics
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier

from hrc_speech_prediction import tree
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
        self.comined_model = None
        self.online = online
        self.X_context, self.context_actions = \
            features.get_context_features(self.data)
        self.X_speech, _ = features.get_bow_features(
            self.data, tfidf=tfidf, n_grams=n_grams)
        # Participants we will exclude from testing, but will train on
        self.test_participants = ["15.ADT"]
        self.lam = lam

        self.C_probs = []
        self.S_probs = []
        self.B_probs = []

        self.Ys = []
        self.Us = []

    def get_Xs(self, indices):
        return self.X_context[indices, :], self.X_speech[indices, :]

    def get_labels(self, indices):
        return [list(self.data.labels)[i] for i in indices]

    def check_indices(self, train_idx, test_idx):
        assert (not set(train_idx).intersection(test_idx))

    def test_on_one_participant(self, data_type="context", speech_eps=0.15):
        """
        Leaves on participant out of training and then tests on them. Does
        this for each participant
        """
        print("Running test on one participant...")
        participants = TRAIN_PARTICIPANTS
        results = {}
        # Get the indices for training and testing data
        for tst in participants:
            # Each sublist contains indices for each trial (a trial being an np array)
            print(list(self.data.data[tst].ids))
            test_idx = [
                [trial.ids for trial in self.data.data[tst]]
            ]
            print(test_idx)
            # a list of lists where the sublist contains
            # the trials for each participant
            train_idx = [
                [
                    trial.ids for trial in self.data.data[part]
                ]
                for part in TRAIN_PARTICIPANTS
            ]
            results[tst] = self._evaluate_on(train_idx, test_idx,
                                             data_type, speech_eps)
        self._print_result_table(results, "Participants")
        return [t[2] for t in results.values()]

    def test_on_one_trial(self, data_type="context", speech_eps=10):
        """
        Excludes on trial from training (i.e. A, B, or C) and uses these
        excluded trials as tests
        """
        print("Running test on one trial...")
        results = {}
        for tst in ['A', 'B', 'C']:
            test_idx = [
                [
                    trial.ids for trial in self.data.data[part]
                    if tst == trial.instruction
                ]
                for part in TRAIN_PARTICIPANTS
            ]
            train_idx = [

                [
                    trial.ids for trial in self.data.data[part]
                    if tst != trial.instruction
                ]
                for part in TRAIN_PARTICIPANTS
                # i for p in TRAIN_PARTICIPANTS for i in self.data.data[p].ids
                # if i not in test_idx
            ]
            results[tst] = self._evaluate_on(train_idx, test_idx,
                                             data_type, speech_eps)
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

    def test_incremental_learning(self, data_type):
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

        print(score)
        xaxis = np.array([i for i in range(0, self.X_context.shape[0] - 1)])
        score = np.array(score)

        xnew = np.linspace(xaxis.min(), xaxis.max(), 1000)
        score_smooth = spline(xaxis, score,xnew)

        plt.plot(xnew, score_smooth)
        plt.show()


    def _evaluate_on(self, train_idx, test_idx, data_type, speech_eps):
        # self.check_indices(train_idx, test_idx)

        flat_train_idx = [i for p in train_idx for t in p for i in t]
        flat_test_idx = [i for p in test_idx for t in p for i in t]
        test_utters = [list(self.data.utterances)[i] for i in flat_test_idx]
        self.Us += [i for i in test_utters]

        context_train  = [self.get_labels(t) for p in train_idx for t in p]
        # Train set
        speech_train_Xs = self.get_Xs(flat_train_idx)
        speech_train_Y  = self.get_labels(flat_train_idx)

        # Test set a list of np arrays
        # each array represents the utterances for one participant
        test_X = [self.get_Xs(t)[1] for p in test_idx for t in p]
        test_Y = self.get_labels(flat_test_idx)
        self.Ys += [i for i in test_Y]
        # Train
        speech_model = self.model(
            self.context_actions,
            features=data_type,
            randomize_context=.25, ).fit(speech_train_Xs[0], speech_train_Xs[1],
                                         speech_train_Y, online=self.online)
        # Evaluate
        self.comined_model = tree.Tree(root=tree.Node(),
                                       speech_model=speech_model)
        # train context model on state visitations
        for t in context_train:
            self.comined_model.add_branch(t)

        return self._paricipant_accuracy_score(test_X, test_Y,
                                               self.comined_model, test_utters)

    def _paricipant_accuracy_score(self, test_X, labels, model, utters):
        combined_model = model
        context_model = tree.Tree(root=combined_model.root,
                                  speech_model=combined_model.speech_model)

        score = np.zeros(3) # speech, context, and both respectively
        n = len(labels)
        for t in test_X:
            # curr_b = model
            # curr_c = model
            # curr_s = model
            print("\nNEW TRIAL\n")
            for u in t:
                y = labels.pop(0)

                both, s, _, b = combined_model.take_action(model="both",
                                                           utter=u,
                                                           return_probs=True)

                speech = combined_model.pred_action(s)
                context, c = context_model.take_action(model="context")

                self.C_probs.append(c)
                self.S_probs.append(s)
                self.B_probs.append(b)

                #print(y, speech, context, both)

                score[0] += 1.0 * (speech in y)
                score[1] += 1.0 * (context in y)
                score[2] += 1.0 * (both in y)

            combined_model.reset()
            context_model.reset()
        return score / n



    def plot_random_probs(self, num_plots):
        self.test_on_one_participant("speech", speech_eps=0.1)
        rand_idx = random.sample([i for i in range(0, len(self.C_probs))], num_plots)

        Cs = np.array(self.C_probs)
        Bs = np.array(self.B_probs)
        Ss = np.array(self.S_probs)


        labels = self.comined_model._speech_model.actions
        width = .3

        #axes[0].legend(('Speech', 'Context', 'Both'))
        for i in rand_idx:
            fig, ax = plt.subplots(1, 1)
            X = np.arange(len(Cs[0]))
            c = Cs[i, :]
            b = Bs[i, :]
            s = Ss[i, :]

            print(c)

            y = self.Ys[i]
            u = self.Us[i]

            nrmlz = 1.0 / sum(b)

            ax.bar(X-width, s, width=width, color='r', align='center')
            ax.bar(X, c, width=width, color='b', align='center', tick_label=labels)
            ax.bar(X+width, b * nrmlz, width=width, color='g', align='center')
            rects = ax.patches
            max_prob = max(b * nrmlz)

            ax.legend(('Speech', 'Context', 'Both'))

            #This draws a star above most probable action
            for r in rects:
                if r.get_height() == max_prob:
                    ax.text(r.get_x() + r.get_width()/2,
                            r.get_height() * 1.01, '*', ha='center', va='bottom', fontsize="12")

            ax.text(self.comined_model._speech_model.actions.index(y), max_prob, "$", fontsize="12")
            #plt.xticks(X, labels, rotation=70)
            ax.set_title(u, fontsize="9")
            plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='right')
            fig.tight_layout()
            plt.savefig("/home/jake/catkin_ws/src/hrc_speech_prediction/figs/fig.{}.png".format(i))
            plt.clf() 
    




    def _print_result_table(self, results, key_title):
        w = max(len(key_title), len("Accuracy"))
        print("{:<{w}}: {} ".format(
            key_title,
            " ".join(["{:^7}".format(k) for k in results.keys()]),
            w=w) +
              "{:^7} {:^7}".format("AVG.", "STD."))
        print("{:<{w}}: {} ".format(
            "Speech ",
            " ".join(["{:^7.2f}".format(v[0]) for v in results.values()]), w=w)
              + self._print_global_results(v[0] for v in results.values())
        )
        print("{:<{w}}: {} ".format(
            "Context",
            " ".join(["{:^7.2f}".format(v[1]) for v in results.values()]), w=w)
              + self._print_global_results(v[1] for v in results.values())
        )
        print("{:<{w}}: {} ".format(
            "Both",
            " ".join(["{:^7.2f}".format(v[2]) for v in results.values()]), w=w)
              + self._print_global_results(v[2] for v in results.values())
        )
        #self._print_global_results(results.values())

    def _print_global_results(self, results):
        r = list(results)
        return "{:.3f} {:.4f}".format(np.average(r), np.std(r))


if __name__ == '__main__':
    args = parser.parse_args()

    ev = EvaluateModel(
        ContextFilterModel.model_generator(SGDClassifier,
                                           loss='log',
                                           average=True,
                                           penalty='l2',
                                           alpha=0.0002),
        args.path,
        lam=0.5,
        n_grams=(1, 2),
        online=False
    )
    ev.test_on_one_participant("speech", speech_eps=0.01)
    #ev.plot_random_probs(3)
    #ev.test_on_one_participant()
