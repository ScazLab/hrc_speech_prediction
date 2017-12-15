import os
import argparse

import numpy as np
from sklearn.externals import joblib

from hrc_speech_prediction import features
from hrc_speech_prediction.data import TrainData, TRAIN_PARTICIPANTS, ALL_ACTIONS


parser = argparse.ArgumentParser("Train and evaluate classifier")
parser.add_argument(
    'path', help='path to the experiment data', default=os.path.curdir)


class Node(object):
    def __init__(self,data_path="../train.json",
                 model_path="../models/model_speech_table.pkl",
                 vectorizer_path ="../models/vocabulary.pkl"):
        "A state in a task trajectory"
        self._count = 1.0
        self._model = joblib.load(model_path)
        self._children = {}  # keys: action taken, val: list of nodes
        # self._children = dict(zip(self._model.actions, 
        #                           [Node() for i in self._model.actions]))
        self._data = TrainData.load(data_path)
        self._X_speech, _ = features.get_bow_features(self._data)
        self._X_context = np.ones((len(self._model.actions)), dtype='bool')
        
        self._vectorizer = joblib.load(vectorizer_path)
        self._eps = 0.15 # Prior on unseen actions


    @property
    def n_children(self):
        return len(self._children)

    @property
    def seen_children(self):
        return self._children.keys()

    @property
    def children_counts(self):
        counts = []
        for c in self.seen_children:
            counts.append(self._children[c]._count)
        return counts

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
            self._children[new_act] = Node()
        self._children[new_act].add_nodes(new_acts)

    def _get_speech_probs(self, utter):
        "Takes a speech utterance and returns probabilities for each \
        possible action on the speech model alone"

        x_u = self._vectorizer.transform([utter])
        pred = self._model._predict_proba(self._X_context[None,:], x_u)[0]
        return dict(zip(self._model.actions,pred))

    def _get_visit_probs(self):
        "Returns probabilities for taking each child action based \
        only on how many times each child has been visited"

        prior_seen = 1 - self._eps  # weight given to seen action
        unseen_denom = 1 / ((self._model.n_actions - self.n_children) * 1.0)
        s = sum(self.children_counts)

        probs = {k: prior_seen * v._count / s
                 for k, v in self._children.items()}
        probs.update({k: (self._eps * unseen_denom)
                      for k in self._model.actions
                      if k not in self.seen_children})
        
        print(sum(probs.values()))
        return probs
                                                                    
    def take_action(self, utter):
        visit_probs = self._get_visit_probs()
        speech_probs = self._get_speech_probs(utter)

        keys = speech_probs.keys()
        product_probs = [visit_probs[k] * speech_probs[k] for k in keys]

        final_probs = dict(zip(keys, product_probs))

        return max(final_probs, key=lambda k:final_probs[k])

    def __str__(self, level=0, val="init"):
        ret ="\t"* level + "{}: {}\n".format(val, self._count)
        for k,v in self._children.items():
            ret += v.__str__(level + 1, k)
        return ret


args = parser.parse_args()
n1 = Node()
n1.add_nodes(["foot_2"])
n1.add_nodes(["foot_2", "foot_1", "leg_1"])
n1.add_nodes(["chair_back", "seat", "back_1"])


print(n1)
print("Predicted action :", n1.take_action("Pass me the last one "))
