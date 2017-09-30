import json
from collections import namedtuple, OrderedDict


# Note on terminology:
# - one session is the full recording of a participant
# - each session consists in several trials each corresponding to
#   an instruction (sheet)


Trial = namedtuple('Trial', ['instruction', 'pairs', 'initial_time'])


class Session(list):

    @property
    def order(self):
        return [t.instruction for t in self]

    def from_instruction(self, instruction):
        return self[self.order.index(instruction)]

    @classmethod
    def deserialize(cls, lst):
        assert(all([len(t) == 3 for t in lst]))
        return cls([Trial(instruction=t[0], pairs=t[1], initial_time=t[2])
                    for t in lst])


class TrainData(object):
    """Contains training data.

    data:
    {paricipant_id: [(trial_instruction, trial_pairs, initial_time)]
    }

    where trial_pairs are: ((action, start_time, end_time),
                            (utterance, start_time, end_time))
    """

    def __init__(self, data):
        self.data = data

    @property
    def n_participants(self):
        return len(self.data)

    def all_trials(self):
        for participant in self.data:
            for x in self.data[participant]:
                yield x

    def count_by_instructions(self):
        counts = {}
        for participant in self.data:
            for i in self.data[participant].order:
                counts[i] = 1 + counts.get(i, 0)
        return counts

    def all_words(self):
        raise NotImplementedError

    def dump(self, path):
        with open(path, 'w') as f:
            json.dump(self.data, f, indent=2)

    @classmethod
    def load(cls, path):
        with open(path) as f:
            data = json.load(f, object_pairs_hook=OrderedDict)
        return cls(OrderedDict([
            (p, Session.deserialize(data[p])) for p in data
        ]))
