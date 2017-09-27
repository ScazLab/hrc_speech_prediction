#!/usr/bin/env python


import os
import math
import argparse

import rospy
import rosbag

from ros_speech2text.msg import event


ACTION_TOPICS = ['/action_provider/{}/state'.format(s)
                 for s in ['left', 'right']]
SPEECH_TOPIC = '/speech_to_text/log'


parser = argparse.ArgumentParser("Displays the bag as a log of relevant information")
parser.add_argument('path', help='path to the experiment files', default=os.path.curdir)
parser.add_argument('participant', help='id of the participant')


class ActionDetector(object):

    """Detects completed actions.

    In particular ignores errored or killed actions. Consolidate actions
    with start and end time.
    """

    WORKING = 0
    DONE = 1
    ERROR = 2

    def __init__(self, callback):
        self.callback = callback  # should accept args: action, t_start, t_end
        self.state = None
        self.current = None  # object name are used to refer to actions
        self.start = None

    def new_state(self, m):
        if m.message.object:
            assert(m.message.action == 'get_pass')
            if m.message.state == 'WORKING':
                if (self.state == self.WORKING and
                        self.current != m.message.object):
                    print('[WARNING] WORKING while working (skipping {}, '
                          'keeping {}).'.format(self.current, m.message.object))
                self.state = self.WORKING
                self.current = m.message.object
                self.start = m.timestamp
            elif m.message.state == 'DONE':
                if self.state != self.WORKING:
                    print('[WARNING] ignoring DONE while not working.')
                else:
                    self.callback(self.current, self.start, m.timestamp)
                    self.state = self.DONE
            elif m.message.state in ('ERROR', 'KILLED'):
                self.state = self.ERROR
            else:
                raise ValueError('Unknown state {}'.format(m.message.state))


class SpeechActionPairer(object):
    """Pairs speech utterances and actions.

    - an action might be associated with several speech utterances,
    - if an action is repeated it is assumed it has failed,
    - an utterance is associated with an action if it starts after the previous
    action has started,
    - the class uses two pass to first sort events by starting time (which is
    not the reporting time) and then pair them.

    TODO: also add a maximum time for a sentence to start before the action it
          is associated to.
    """

    ACTION = 0
    UTTERANCE = 1

    def __init__(self):
        self.events = []

    def new_action(self, action, t_start, t_end):
        self.events.append((self.ACTION, action, t_start, t_end))

    def new_utterance(self, utterance, t_start, t_end):
        self.events.append((self.UTTERANCE, utterance, t_start, t_end))

    def get_associations(self):
        events = sorted(self.events, key=lambda e: e[2])
        prev_action = (None, 0, 0)
        prev_utterances = []
        cur_utterances = []
        for e in events:
            if e[0] == self.ACTION:
                if e[1] == prev_action[0]:
                    prev_action = e[1:]  # update action to keep time of
                    # the last observed trial (the previous must have failed)
                    prev_utterances.extend(cur_utterances)
                else:
                    if prev_action[0] is not None:
                        yield (prev_action, prev_utterances)
                    prev_utterances = cur_utterances
                    prev_action = e[1:]
                cur_utterances = []
            elif e[0] == self.UTTERANCE:
                cur_utterances.append(e[1:])
        yield (prev_action, prev_utterances)


def participant_bags(data_path, participant):
    participant_dir = os.path.expanduser(os.path.join(data_path, participant))
    files = sorted(os.listdir(participant_dir))
    for f in files:
        if f.endswith('.bag'):
            yield rosbag.Bag(os.path.join(participant_dir, f))


def bag_summary(bag):
    for m in bag.read_messages():
        pretty_print(m, start_time=bag.get_start_time())


def pretty_print(m, start_time=rospy.Time(0)):
    if m.topic in ACTION_TOPICS and m.message.object:
        print("{}[{}] {} {} ({})".format(
            format_message_time(m, start_time), m.message.state, m.message.action, m.message.object,
            m.topic.split('/')[-2]))
    elif m.topic in SPEECH_TOPIC and m.message.event == event.DECODED:
        print("{}\"{}\" ({} -> {})".format(
            format_message_time(m, start_time),
            m.message.transcript.transcript.strip(),
            format_time(m.message.transcript.start_time, start_time),
            format_time(m.message.transcript.end_time, start_time)))


def format_message_time(message, start_time):
    return "[{:>7s}] ".format(format_time(message.timestamp, start_time))


def format_time(time, start_time):
    delta = time.to_sec() - start_time
    minutes = math.floor(delta / 60)
    return "{:.0f}:{:04.1f}".format(minutes, delta - 60 * minutes)


def parse_bag(bag):
    pairer = SpeechActionPairer()
    left_detector = ActionDetector(pairer.new_action)
    right_detector = ActionDetector(pairer.new_action)
    for m in bag.read_messages():
        pretty_print(m, start_time=bag.get_start_time())
        if m.topic == ACTION_TOPICS[0]:
            left_detector.new_state(m)
        elif m.topic == ACTION_TOPICS[1]:
            right_detector.new_state(m)
        elif m.topic in SPEECH_TOPIC and m.message.event == event.DECODED:
            pairer.new_utterance(m.message.transcript.transcript,
                                 m.message.transcript.start_time,
                                 m.message.transcript.end_time)
    return pairer


if __name__ == "__main__":

    args = parser.parse_args()

    for b in participant_bags(args.path, args.participant):
        print('\nNew Session: {} ({})\n'.format(args.participant,
                                                b.filename.split('/')[-1]))
        print('Topics:')
        types, topics = b.get_type_and_topic_info()
        for t in topics:
            print('- {}: {} messages of type {}'.format(
                t, topics[t].message_count, topics[t].msg_type))
        pairer = parse_bag(b)
        print('\nAssociation:')
        for a, u in pairer.get_associations():
            print(a[0], ' | '.join([uu[0] for uu in u]))
