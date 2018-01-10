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


# TODO Make Node more efficient
# TODO cache calculated probabilities
# TODO cache speech_model
class Node(object):
    def __init__(self,
                 model,
                 data=TrainData.load("../train.json"),
                 vectorizer=joblib.load("../models/vocabulary.pkl")):
        "A state in a task trajectory"
        self._count = 1.0
        self._speech_model = model
        self._children = {}  # keys: action taken, val: list of nodes
        # self._children = dict(zip(self._speech_model.actions, 
        #                           [Node() for i in self._speech_model.actions]))
        self._data = data
        #self._X_speech, _ = features.get_bow_features(self._data)
        self._X_context = np.ones((len(self._speech_model.actions)), dtype='bool')
        
        self._vectorizer = vectorizer
        self._eps = 0.15 # Prior on unseen actions


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

    def add_nodes(self, new_acts):
        if not new_acts:
            return self
        new_act = new_acts.pop(0)
        if new_act in self.seen_children:
            self._children[new_act]._increment_count()
        else:
            self._children[new_act] = Node(model=self._speech_model,
                                           data=self._data,
                                           vectorizer=self._vectorizer)
        self._children[new_act].add_nodes(new_acts)

    def _get_speech_probs(self, utter):
        "Takes a speech utterance and returns probabilities for each \
        possible action on the speech model alone"

        if isinstance(utter, str):
            x_u = self._vectorizer.transform([utter])
        else:
            x_u = utter # Then the input is an numpy array already
        pred = self._speech_model._predict_proba(self._X_context[None,:], x_u)[0]
        return pred
        #return dict(zip(self._speech_model.actions, pred))

    def _get_visit_probs(self):
        "Returns probabilities for taking each child action based \
        only on how many times each child has been visited"

        prior_s = 1 - self._eps  # weight given to seen action
        unseen = 1 / ((self._speech_model.n_actions - self.n_children) * 1.0)
        s = 1.0 / (sum(self.children_counts) + .00001)


        return np.array(
            [prior_s * self._children[k]._count * s # prob for seen actions from current state
             if k in self.seen_children
             else self._eps * unseen # prob for unseen
             for k in self._speech_model.actions]
        )

    def _get_next_action(self, action):
        try:
            return self._children[action]
        except KeyError:
            print("ERROR: Action hasn't been taken from current state.")
            return Node(model=self._speech_model,
                        data=self._data,
                        vectorizer=self._vectorizer)

    def take_action(self, utter, plot=False):
        visit_probs = self._get_visit_probs()
        speech_probs = self._get_speech_probs(utter)

        final_probs = np.multiply(visit_probs, speech_probs)
        #{k:visit_probs[k] * speech_probs[k] for k in keys}

        if plot:
            self.plot_predicitions(speech_probs, visit_probs, final_probs)

        # next_act = max(final_probs, key=final_probs.get)
        next_act = self._speech_model.actions[np.argmax(final_probs)]

        return self._get_next_action(next_act), next_act



    def plot_predicitions(self, speech, context, both):
        "Plots the probabilities for each possible action provided by speech, \
        context, and speech + context "
        X = np.arange(len(both))
        ax = plt.subplot(111)

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

        plt.xticks(X, self._speech_model.actions, rotation=70)
        plt.title("Next action to take")
        plt.show()


    def __str__(self, level=0, val="init"):
        ret ="\t"* level + "{}: {}\n".format(val, self._count)
        for k,v in self._children.items():
            ret += v.__str__(level + 1, k)
        return ret


if __name__ == '__main__':
    speech_model = joblib.load("../models/model_speech_table.pkl")
    args = parser.parse_args()
    n1 = Node(speech_model)
    n1.add_nodes(["foot_2"])
    n1.add_nodes(["front_2"])
    n1.add_nodes(["foot_2", "foot_1", "leg_1"])
    n1.add_nodes(["chair_back", "seat", "back_1"])


    print(n1)
    print("Predicted action :", n1.take_action("pass the blue piece with two red stripes at the bottom", plot=True))
