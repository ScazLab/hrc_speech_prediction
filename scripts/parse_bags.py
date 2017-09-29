#!/usr/bin/env python

import os
import json
import argparse

from hrc_speech_prediction.bag_parsing import participant_bags, parse_bag


parser = argparse.ArgumentParser(
    "Reads bags from experiments and export training as json")
parser.add_argument('path', help='path to the experiment files', default=os.path.curdir)


SESSIONS = {
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


def session_to_dict(path, participant, instructions):
    files = []
    associations = {}
    bags = participant_bags(path, participant)
    for b, instr in zip(bags, instructions):
        files.append(b.filename)
        print('\nLoading: {} ({}: {})'.format(participant, instr,
                                              os.path.split(b.filename)[-1]))
        pairer = parse_bag(b)
        pairs = list(pairer.get_associations())
        associations.get(instr, []).extend(pairs)  # Append if split bag
        print("Total: {} actions found with {} non-empty utterances.".format(
            len(pairs), sum([len(u) > 0 for a, u in pairs])))
    try:
        next(bags)
    except StopIteration:
        pass
    else:
        raise ValueError('Too many bag files for {} (expected {})'.format(
            participant, len(bags)))
    return {'instructions': instructions,
            'associations': pairs,
            'sources': files,
            }


if __name__ == "__main__":

    args = parser.parse_args()

    data = {p: session_to_dict(args.path, p, SESSIONS[p])
            for p in SESSIONS}
    all_instructions = [i for session in data
                        for i in data[session]['instructions']]
    print("\nTotal for instructions: " +
          ", ".join(["{}: {}".format(x, all_instructions.count(x))
                     for x in ['A', 'B', 'C']])
          )
    out_path = os.path.join(args.path, 'train.json')
    with open(out_path, 'w') as f:
        json.dump(data, f, indent=2)
    print("Written: {} sessions to {}".format(len(SESSIONS), out_path))
