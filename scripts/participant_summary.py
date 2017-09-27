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
        print("{}[{}] {} on {} ({})".format(
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
        bag_summary(b)
