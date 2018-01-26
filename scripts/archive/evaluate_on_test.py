import os
import json
import argparse
from collections import OrderedDict

import numpy as np
from sklearn.externals import joblib

from hrc_speech_prediction.data import ALL_ACTIONS
from hrc_speech_prediction.defaults import DATA_PATH


parser = argparse.ArgumentParser("Evaluate classifier on test data")
parser.add_argument('path', help='path to the experiment data',
                    default=os.path.curdir)
parser.add_argument('-m', '--model', help='model to use',
                    choices=['speech', 'both', 'speech_table', 'both_table'],
                    default='both')
args = parser.parse_args()


def remove_pairs(l):
    return [x for i, x in enumerate(l) if i == 0 or not x == l[i - 1]]


def trial_to_context(trial):
    X = np.ones((len(trial), len(ALL_ACTIONS)))
    for i, pair in enumerate(trial):
        X[:i, ALL_ACTIONS.index(pair[0])] = 1  # in case action failed before
        # mark missing object after it's taken
        X[(i + 1):, ALL_ACTIONS.index(pair[0])] = 0
    return X


args = parser.parse_args()
model = joblib.load(os.path.join(args.path, "model_{}.pkl".format(args.model)))
vectorizer = joblib.load(os.path.join(args.path, "vocabulary.pkl"))

with open(os.path.join(DATA_PATH, 'test_data.json')) as f:
    d = json.load(f)
    d = OrderedDict([(p, d[p]) for p in sorted([pp for pp in d])])

X_c = np.concatenate([trial_to_context(t) for p in d for t in d[p]['trials']],
                     axis=0)
X_s = vectorizer.transform([pair[1] for p in d
                            for t in d[p]['trials']
                            for pair in t])
labels = [pair[0] for p in d for t in d[p]['trials'] for pair in t]
indexes = [pair[0] for p in d for t in d[p]['trials'] for pair in t]

participant = [p for p in d for t in d[p]['trials'] for pair in t]
instruction = [i for p in d for t, i in zip(d[p]['trials'], 'AET') for pair in t]

prediction = model.predict(X_c, X_s)

per_participant = [
    np.average([pred == lab
                for p, pred, lab in zip(participant, prediction, labels)
                if p == part])
    for part in d]
per_instruction = [
    np.average([pred == lab
                for i, pred, lab in zip(instruction, prediction, labels)
                if i == instr])
    for instr in 'AET']
per_trial = [
    [np.average([pred == lab
                 for p, i, pred, lab in zip(participant, instruction, prediction, labels)
                 if i == instr and p == part])
     for instr in 'AET']
    for part in d]
print('\nResults for model {}:'.format(args.model))
print('     {:^5} {:^5} {:^5} | {:^5}'.format(*'AET='))
fmt = '{:<3} ' + ' {:^4.3f}' * 3
for i, p in enumerate(d):
    print((fmt + ' | {:^4.3f}').format(
        p, per_trial[i][0], per_trial[i][1], per_trial[i][2], per_participant[i]))
print(fmt.format('==', per_instruction[0], per_instruction[1], per_instruction[2]))
print(np.average([p == l for p, l in zip(prediction, labels)]))
