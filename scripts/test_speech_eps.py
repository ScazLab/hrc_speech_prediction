import os
import argparse

import random as r
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import SGDClassifier

from hrc_speech_prediction import (nodes, evaluate_tree_model)
from hrc_speech_prediction.models import (BaseModel,
                                          ContextFilterModel,
                                          PragmaticModel)

parser = argparse.ArgumentParser("Train and evaluate classifier")
parser.add_argument(
    'path', help='path to the experiment data', default=os.path.curdir)

class SimModel(evaluate_tree_model.EvaluateModel):
    def __init__(self,
                 model,
                 data_path,
                 lam=1,
                 n_grams=(1, 1),
                 tfidf=False,
                 online=False):
        super.__init__(model, data_path, n_grams, tfidf, online)

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
    
    def _evaluate_on(self, train_idx, test_idx, data_type, speech_eps):
        # self.check_indices(train_idx, test_idx)

        flat_train_idx = [i for p in train_idx for t in p for i in t]
        flat_test_idx = [i for p in test_idx for t in p for i in t]
        test_utters = [list(self.data.utterances)[i] for i in flat_test_idx]

        context_train  = [self.get_labels(t) for p in train_idx for t in p]
        # Train set
        speech_train_Xs = self.get_Xs(flat_train_idx)
        speech_train_Y  = self.get_labels(flat_train_idx)

        # Test set a list of np arrays
        # each array represents the utterances for one participant
        test_X = [self.get_Xs(t)[1] for p in test_idx for t in p]
        test_Y = self.get_labels(flat_test_idx)
        # Train
        speech_model = self.model(
            self.context_actions,
            features=data_type,
            randomize_context=.25, ).fit(speech_train_Xs[0], speech_train_Xs[1],
                                         speech_train_Y, online=self.online)
        # Evaluate
        context_model = nodes.Node(model=speech_model, speech_eps=speech_eps)
        # train context model on state visitations
        for t in context_train:
            context_model.add_nodes(t)

        num_actions = context_model._speech_model.actions
        return self._paricipant_accuracy_score(test_X, test_Y, context_model,
                                               test_utters, num_actions)

    def _paricipant_accuracy_score(self, test_X, labels, model, utters, num_actions):
        score = np.zeros(3) # speech, context, and both respectively
        s = len(labels)

        Cs = np.zeros(shape=(len(labels), len(num_actions)))
        Ss = np.zeros(shape=(len(labels), len(num_actions)))
        Bs = np.zeros(shape=(len(labels), len(num_actions)))

        ys = []

        prob = 1.0 / s # use this to determine which trial to graph
        for t in test_X:
            curr_b = model
            curr_c = model
            curr_s = model
            print("\nNEW TRIAL\n")
            for u in t:
                y = labels.pop(0)

                curr_b, both = curr_b.take_action(u,
                                                  plot=False,
                                                  pred_type='both',
                                                  real_utter=utters.pop(0),
                                                  actual=y)

def plot_speech_eps_results():
    args = parser.parse_args()

    ev = evaluate_tree_model.EvaluateModel(
        ContextFilterModel.model_generator(SGDClassifier,
                                           loss='log',
                                           average=True,
                                           penalty='l1'),
        args.path,
        lam=0.5,
        n_grams=(1, 2),
        online=False
    )

    xs = np.array([.001, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100, 1000])

    ev.test_on_one_participant("speech", 0.01)

    scores = np.array(
        [np.average(ev.test_on_one_participant("speech", x)) for x in xs]
    )

    #ax = plt.subplot(111)
    plt.plot(xs, scores, "o")
    plt.xlabel("eps_speech / eps_context")
    plt.xscale('log')
    plt.title("speech eps by score")
    plt.show()

def plot_ten_predictions():
    args = parser.parse_args()

    ev = evaluate_tree_model.EvaluateModel(
        ContextFilterModel.model_generator(SGDClassifier,
                                           loss='log',
                                           average=True,
                                           penalty='l2'),
        args.path,
        lam=0.5,
        n_grams=(1, 2),
        online=False
    )

    

#plot_speech_eps_results()
