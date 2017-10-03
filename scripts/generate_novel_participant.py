import os
import json
import argparse

from hrc_speech_prediction import data

parser = argparse.ArgumentParser("Create two new trials from 11.BCA's data")
parser.add_argument('path', help='path to the experiment data',
                    default=os.path.curdir)


d_actions  = ["leg_7", "top_2", "top_1", "chair_back", "screwdriver_1",
              "foot_3", "leg_3", "foot_4", "leg_4", "foot_1", "leg_1",
              "foot_2", "leg_2", "front_1", "front_3", "back_1", "back_2",
              "seat", "leg_5", "leg_6"]

e_actions = ["screwdriver_1", "foot_2", "leg_2", "foot_4", "leg_4", "foot_3",
             "leg_3", "foot_1", "leg_1", "back_1", "back_2", "front_1",
             "front_3", "seat", "leg_5", "leg_6", "leg_7", "top_2",
             "top_1", "chair_back"]


with open(os.path.join(os.path.dirname(__file__), 'P11.json')) as f:
    d = json.load(f)

A = dict(d['A'])
B = dict(d['B'])
C = dict(d['C'])


def get_pairs(actions):
    t_actions = [(a, 0, 0) for a in actions]
    t_utterances = []

    for a in actions:
        t_u = []
        for u in B[a]:
            t_u.append((u, 0, 0))
        t_utterances.append(t_u)

    return zip(t_actions, t_utterances)


trials = [data.Trial('D', get_pairs(d_actions), 0),
          data.Trial('E', get_pairs(e_actions), 0)]

new_data = data.TrainData({'DE': data.Session(trials)})
args = parser.parse_args()
new_data.dump(os.path.join(args.path, "new_participant.json"))
