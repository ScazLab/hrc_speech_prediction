import os
import math

import rospy
import rosbag

from ros_speech2text.msg import event


ACTION_TOPICS = ['/action_provider/{}/state'.format(s)
                 for s in ['left', 'right']]
SPEECH_TOPIC = '/speech_to_text/log'


def _relative_time(rospy_time, start_time):
    return rospy_time.to_sec() - start_time


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
            if m.message.action != 'get_pass':
                raise ValueError(
                    "Incorrect action: '{}': expecting 'get_pass'".format(
                        m.message.action))
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
                    self._end_action(m)
            elif m.message.state in ('ERROR', 'KILLED'):
                self.state = self.ERROR
            else:
                raise ValueError('Unknown state {}'.format(m.message.state))
        elif m.message.action == 'home':
            if self.current is not None:
                self._end_action(m)

    def _end_action(self, m):
        self.callback(self.current, self.start, m.timestamp)
        self.state = self.DONE


class SpeechActionPairer(object):
    """Pairs speech utterances and actions.

    - an action might be associated with several speech utterances,
    - if an action is repeated it is assumed it has failed,
    - an utterance is associated with an action if it starts after the previous
    action has started,
    - the class uses two pass to first sort events by starting time (which is
    not the reporting time) and then pair them,
    - also returns relative times as floats in seconds.

    TODO: also add a maximum time for a sentence to start before the action it
          is associated to.
    """

    ACTION = 0
    UTTERANCE = 1

    def __init__(self, initial_time):
        self.events = []
        self.initial_time = initial_time

    def new_action(self, action, t_start, t_end):
        self.events.append((self.ACTION, action, t_start, t_end))

    def new_utterance(self, utterance, t_start, t_end):
        self.events.append((self.UTTERANCE, utterance, t_start, t_end))

    def get_associations(self):
        events = sorted(self.events, key=lambda e: e[2])
        prev_action = (None, -1., -1.)
        prev_utterances = []
        cur_utterances = []
        for e in events:
            if e[0] == self.ACTION:
                if e[1] == prev_action[0]:
                    prev_action = self._times_to_relative(e[1:])
                    # update action to keep time of the last observed trial
                    # (the previous must have failed)
                    prev_utterances.extend(cur_utterances)
                else:
                    if prev_action[0] is not None:
                        yield (prev_action, prev_utterances)
                    prev_utterances = cur_utterances
                    prev_action = self._times_to_relative(e[1:])
                cur_utterances = []
            elif e[0] == self.UTTERANCE:
                cur_utterances.append(self._times_to_relative(e[1:]))
        yield (prev_action, prev_utterances)

    def _times_to_relative(self, timed_action_or_utterance):
        a_or_u, t_start, t_end = timed_action_or_utterance
        return (a_or_u,
                _relative_time(t_start, self.initial_time),
                _relative_time(t_end, self.initial_time))


def guess_participant_id(data_path, prefix):
    files = sorted(os.listdir(os.path.expanduser(data_path)))
    candidates = [f for f in files if (
        f.startswith(prefix) and os.path.isdir(os.path.join(data_path, f)))]
    if len(candidates) == 0:
        raise FileNotFoundError("Could not find any file starting with {} at "
                                "the given path.".format(prefix))
    else:
        return candidates[0]


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
    delta = _relative_time(time, start_time)
    minutes = math.floor(delta / 60)
    return "{:.0f}:{:04.1f}".format(minutes, delta - 60 * minutes)


def parse_bag(bag):
    pairer = SpeechActionPairer(bag.get_start_time())
    left_detector = ActionDetector(pairer.new_action)
    right_detector = ActionDetector(pairer.new_action)
    for m in bag.read_messages():
        if m.topic == ACTION_TOPICS[0]:
            left_detector.new_state(m)
        elif m.topic == ACTION_TOPICS[1]:
            right_detector.new_state(m)
        elif m.topic in SPEECH_TOPIC and m.message.event == event.DECODED:
            pairer.new_utterance(m.message.transcript.transcript,
                                 m.message.transcript.start_time,
                                 m.message.transcript.end_time)
    return pairer
