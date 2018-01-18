import os
import argparse

import numpy as np
import matplotlib.pyplot as plt
from sklearn.externals import joblib

from hrc_speech_prediction import features
from hrc_speech_prediction.data import (TrainData,
                                        TRAIN_PARTICIPANTS, ALL_ACTIONS)


parser = argparse.ArgumentParser("Train and evaluate classifier")
parser.add_argument(
    'path', help='path to the experiment data', default=os.path.curdir)


class Tree(object):
    def __init__(self, speech_model, root,
                 vectorizer=joblib.load("../models/vocabulary.pkl"),
                 speech_eps=0.15, context_eps=0.15):
        self.speech_model = speech_model
        self._vectorizer = vectorizer

        self._speech_eps = speech_eps
        self._context_eps = context_eps

        self.root = root
        self._curr = self.root # current node we are looking at

        self._X_context = np.ones((len(self.speech_model.actions)), dtype='bool')
        self.actions = self.speech_model.actions


    @property
    def curr(self):
        return self._curr

    @curr.setter
    def curr(self, node):
        self._curr = node

    def add_branch(self, list_of_actions):
        return self.root.add_branch(list_of_actions)

    def get_speech_probs(self, utter):
        "Takes a speech utterance and returns probabilities for each \
            possible action on the speech model alone"


        if isinstance(utter, str):
            x_u = self._vectorizer.transform([utter])
        else:
            x_u = utter # Then the input is an numpy array already

        probs = self.speech_model._predict_proba(self._X_context[None,:],
                                                    x_u)[0]

        return self._apply_eps(self._speech_eps, probs)

    def get_context_probs(self):
        probs = self.curr._get_context_probs(self._context_eps,
                                             self.actions)

        return self._apply_eps(self._context_eps, probs)

    def _apply_eps(self, eps, p):
        u = np.array([1.0 / len(self.actions) for i in self.actions])

        return (1.0 - eps) * p + (u * eps)

    def pred_action(self, probs):
        return self.actions[np.argmax(probs)]

    def take_action(self, model="both", utter=None,
                    plot=False, return_probs=False):

        context_probs = self.get_context_probs()
        if model == "context":
            probs = self.get_context_probs()
        elif model == "speech" and utter is not None:
            probs = self.get_speech_probs(utter)
        elif model == "both" and utter is not None:
            context_probs = self.get_context_probs()
            speech_probs = self.get_speech_probs(utter)

            probs = np.multiply(context_probs, speech_probs)

        action = self.pred_action(probs)
        self.curr = self.curr._get_next_node(action)

        if plot:
            self.plot_predicitions(speech_probs, context_probs,
                                   combined_probs, utter)
        if return_probs and model=="both":
            return action, speech_probs, context_probs, probs
        else:
            return action, probs

    def reset(self):
        self._curr = self.root

    def plot_predicitions(self, speech, context, both, utter,
                          actual=None, save_path=None):
        "Plots the probabilities for each possible action provided by speech, \
        context, and speech + context "
        X = np.arange(len(both))
        fig, ax = plt.subplots(nrows=1, ncols=1)

        # Want to normalize 'both' probs for easier visual comparison
        nrmlz = 1.0 / sum(both)

        ax.bar(X-0.2, speech, width=0.2, color='r', align='center')
        ax.bar(X, context, width=0.2, color='b', align='center')
        ax.bar(X+0.2, both * nrmlz, width=0.2, color='g', align='center')

        ax.legend(('Speech', 'Context', 'Both'))

        rects = ax.patches
        max_prob = max(both * nrmlz)

        #This draws a star above most probable action
        for r in rects:
            if r.get_height() == max_prob:
                ax.text(r.get_x() + r.get_width()/2,
                        r.get_height() * 1.01, '*', ha='center', va='bottom')

        if actual:
            ax.text(self.speech_model.actions.index(actual), max_prob, "$")

        plt.xticks(X, self.speech_model.actions, rotation=70)
        plt.title(utter)

        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()

    def __str__(self):
        return self.root.__str__()


class Node(object):
    def __init__(self):
        "A state in a task trajectory"
        self._count = 1.0
        self._children = {}  # keys: action taken, val: list of nodes


    @property
    def n_children(self):
        return len(self._children)

    @property
    def seen_children(self):
        return self._children.keys()

    @property
    def children_counts(self):
        return [self._children[c]._count for c in self.seen_children]

    def _increment_count(self):
        self._count += 1.0
        return self

    def add_branch(self, new_acts):
        if not new_acts:  # list is empty
            return self

        new_act = new_acts.pop(0)

        if new_act in self.seen_children:
            self._children[new_act]._increment_count()
        else:
            self._children[new_act] = Node() 
        self._children[new_act].add_branch(new_acts)


    def _get_context_probs(self, eps, actions):
        "Returns probabilities for taking each child action based \
        only on how many times each child has been visited"

        s = 1.0 / (sum(self.children_counts) + .00001)

        return np.array(
            [self._children[k]._count * s
             if k in self.seen_children else 0.0
             for k in actions]
        )

    def _get_next_node(self, action):
        "Returns the node corresponding to action"
        try:
            return self._children[action] 
        except KeyError:
            print("ERROR: Action hasn't been taken from current state.")
            self._children[action] = Node()
            return self._children[action]


    def take_action(self, utter, plot=False, pred_type="both",
                    real_utter=None, actual=None, return_probs=False):
        visit_probs = self._get_context_probs()
        speech_probs = self._get_speech_probs(utter)

        both_probs = np.multiply(visit_probs, speech_probs)


        if pred_type == "both":
            pred = self.speech_model.actions[np.argmax(both_probs)]

        elif pred_type == "speech":
            pred = self.speech_model.actions[np.argmax(speech_probs)]
        elif pred_type == "context":
            pred = self.speech_model.actions[np.argmax(visit_probs)]

        next_act, ooc = self._get_next_node(pred)
        if plot and pred_type == "both":
            self.plot_predicitions(speech_probs, visit_probs,
                                   both_probs, real_utter, actual)
        if return_probs:
            return (next_act, pred, visit_probs, speech_probs, both_probs)
        else:
            return (next_act, pred)





    def __str__(self, level=0, val="init"):
        ret ="\t"* level + "{}: {}\n".format(val, self._count)
        for k,v in self._children.items():
            ret += v.__str__(level + 1, k)
        return ret


if __name__ == '__main__':
    speech_model = joblib.load("../models/model_speech_table.pkl")
    args = parser.parse_args()
    t = Tree(speech_model=speech_model, root=Node())
    t.add_branch(["foot_2"])
    t.add_branch(["top_1"])
    t.add_branch(["foot_2", "foot_1", "leg_1"])
    t.add_branch(["chair_back", "seat", "back_1"])

    print(t)

    print(t.take_action(utter="pass the blue piece with two red", model="speech", plot=True))
