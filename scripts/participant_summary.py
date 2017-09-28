#!/usr/bin/env python

import os
import argparse

from hrc_speech_prediction.bag_parsing import (participant_bags, guess_participant_id,
                                               parse_bag)


parser = argparse.ArgumentParser("Displays the bag as a log of relevant information")
parser.add_argument('path', help='path to the experiment files', default=os.path.curdir)
parser.add_argument('participant', help='id of the participant')


if __name__ == "__main__":

    args = parser.parse_args()
    participant = guess_participant_id(args.path, args.participant)

    for b in participant_bags(args.path, participant):
        print('\nNew Session: {} ({})\n'.format(args.participant,
                                                b.filename.split('/')[-1]))
        print('Topics:')
        types, topics = b.get_type_and_topic_info()
        for t in topics:
            print('- {}: {} messages of type {}'.format(
                t, topics[t].message_count, topics[t].msg_type))
        pairer = parse_bag(b)
        print('\nAssociation:')
        pairs = list(pairer.get_associations())
        for a, u in pairs:
            print(a[0], ' | '.join([uu[0] for uu in u]))
        print("Total: {} actions found with {} non-empty utterances.".format(
            len(pairs), sum([len(u) > 0 for a, u in pairs])))
