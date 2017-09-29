#!/usr/bin/env python

import os
import argparse

from hrc_speech_prediction.data import TrainData
from hrc_speech_prediction.bag_parsing import participant_bags, parse_bag


parser = argparse.ArgumentParser(
    "Reads bags from experiments and export training as json")
parser.add_argument('path', help='path to the experiment files', default=os.path.curdir)


PARTICIPANTS = {
    '1.ABC':  ['B', 'C'],
    '2.BCA':  ['B', 'C'],  # , 'A'], # for some reason last bag is not readable
    '3.CAB':  ['C', 'A', 'B'],
    '4.ABC':  ['A', 'B', 'C'],
    '5.BCA':  ['B', 'C', 'A'],
    '6.CAB':  ['C', 'A', 'B'],
    '7.ABC':  ['A', 'B'],
    '8.BCA':  ['B', 'B', 'C'],  # First split in two. Only two.
    '9.CAB':  ['C', 'A', 'B'],
    '10.ABC': ['A', 'B', 'C'],
    '11.BCA': ['A'],
}


# Data clean functions

def merge_duplicate_in_8(data):
    # BBC -> BC
    P8 = '8.BCA'
    session = data.data[P8]
    assert(session[0][0] == 'B' and session[1][0] == 'B')
    delta = session[1][2] - session[0][2]  # difference in start time

    def update_times(x, ts, te):
        return (x, ts + delta, te + delta)

    # Add time difference to actions and utterances in second bag
    new_pairs = [
        (update_times(*action), [update_times(*u) for u in utterances])
        for action, utterances in session[1][1]
    ]
    # Merge two first bags
    data.data[P8] = [('B', session[0][1] + new_pairs, session[0][2]),
                     session[2]]


def rename_wrong_actions_in_first_sessions(data):
    """front_2 | foot_5 -> front_3"""

    def rename(action):
        if action in ('front_2', 'foot_5'):
            return 'front_3'
        else:
            return action

    def rename_pair(action, utterances):
        return rename(action), utterances

    for participant in data.data:
        data.data[participant] = [
            (instr, [rename_pair(*pair) for pair in pairs], time)
            for instr, pairs, time in data.data[participant]
        ]


CLEAN_FUNCTIONS = [
    merge_duplicate_in_8,
    rename_wrong_actions_in_first_sessions,
]


# Extracting helper

def participant_to_list(path, participant, instructions):
    associations = []
    bags = participant_bags(path, participant)
    for b, instr in zip(bags, instructions):
        print('\nLoading: {} ({}: {})'.format(participant, instr,
                                              os.path.split(b.filename)[-1]))
        pairer = parse_bag(b)
        pairs = list(pairer.get_associations())
        associations.append((instr, pairs, b.get_start_time()))
        print("Total: {} actions found with {} non-empty utterances.".format(
            len(pairs), sum([len(u) > 0 for a, u in pairs])))
    try:
        next(bags)
    except StopIteration:
        pass
    else:
        raise ValueError('Too many bag files for {} (expected {})'.format(
            participant, len(bags)))
    return associations


if __name__ == "__main__":

    args = parser.parse_args()

    data = TrainData({p: participant_to_list(args.path, p, PARTICIPANTS[p])
                      for p in PARTICIPANTS})

    for fun in CLEAN_FUNCTIONS:
        fun(data)

    counts = data.count_by_instructions()
    print("\nTotal for instructions: " +
          ", ".join(["{}: {}".format(x, counts[x]) for x in counts])
          )
    out_path = os.path.join(args.path, 'train.json')
    data.dump(out_path)
    print("Written: {} participants to {}".format(data.n_participants, out_path))
