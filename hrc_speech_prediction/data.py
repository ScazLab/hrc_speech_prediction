import json
from bisect import bisect
from itertools import chain
from collections import namedtuple, OrderedDict

import numpy as np


# Note on terminology:
# - one session is the full recording of a participant
# - each session consists in several trials each corresponding to
#   an instruction (sheet)


def cumsum_from_0(iterable):
    return [0] + list(np.cumsum(iterable)[:-1])


def bisect_start_id(start_ids, i):
    """bisect a list of starting ids for contiguous subsets and return
       in which subset i belongs.
    """
    return bisect(start_ids, i) - 1


_Trial = namedtuple('Trial', ['instruction', 'pairs', 'initial_time'])


class Trial(_Trial):

    def __init__(self, *args, **kwargs):
        super(Trial, self).__init__(*args, **kwargs)
        self._first_id = None

    @property
    def n_samples(self):
        return len(self.pairs)

    @property
    def utterances(self):
        return [' '.join([u[0] for u in pair[1]]) for pair in self.pairs]

    @property
    def labels(self):
        return [pair[0][0] for pair in self.pairs]

    def set_first_id(self, i):
        self._first_id = i

    @property
    def first_id(self):
        if self._first_id is None:
            raise ValueError('First id not set.')
        return self._first_id

    @property
    def ids(self):
        return [self.first_id + i for i, _ in enumerate(self.pairs)]

    def get_pair_from_id(self, i):
        return self.pairs[i - self.first_id]


class Session(list):

    @property
    def n_samples(self):
        return sum([trial.n_samples for trial in self])

    @property
    def order(self):
        return [t.instruction for t in self]

    @property
    def utterances(self):
        return chain.from_iterable([trial.utterances for trial in self])

    @property
    def labels(self):
        return chain.from_iterable([trial.labels for trial in self])

    def set_first_id(self, i):
        id_deltas = cumsum_from_0([trial.n_samples for trial in self])
        for d, trial in zip(id_deltas, self):
            trial.set_first_id(i + d)

    @property
    def first_id(self):
        return self[0].first_id

    @property
    def ids(self):
        return chain.from_iterable([trial.ids for trial in self])

    def get_pair_from_id(self, i):
        trial_idx = bisect_start_id([trial.first_id for trial in self], i)
        return self[trial_idx].get_pair_from_id(i)

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
        self.reset_ids()

    def guess_participant(self, prefix, first_on_multiple=False):
        candidates = [p for p in self.participants
                      if p.startswith(prefix)]
        if len(candidates) == 0:
            raise ValueError('Cannot find participant with name starting in ' +
                             prefix)
        elif not first_on_multiple and len(candidates) > 1:
            raise ValueError('Several participants start with ' + prefix)
        else:
            return candidates[0]

    def reset_ids(self):
        first_ids = cumsum_from_0([self.data[p].n_samples for p in self.data])
        for i, p in zip(first_ids, self.data):
            self.data[p].set_first_id(i)

    @property
    def participants(self):
        return [p for p in self.data]

    @property
    def n_participants(self):
        return len(self.data)

    @property
    def n_samples(self):
        return sum([self.data[p].n_samples for p in self.data])

    @property
    def ids(self):
        return chain.from_iterable([self.data[p].ids for p in self.data])

    @property
    def utterances(self):
        return chain.from_iterable([self.data[p].utterances for p in self.data])

    @property
    def labels(self):
        return chain.from_iterable([self.data[p].labels for p in self.data])

    def get_pair_from_id(self, i):
        session_idx = bisect_start_id([self.data[p].first_id
                                       for p in self.data], i)
        return self.data[self.participants[session_idx]].get_pair_from_id(i)

    def all_trials(self):
        for participant in self.data:
            for x in self.data[participant]:
                yield x

    def all_words(self):
        raise NotImplementedError

    def count_by_instructions(self):
        counts = {}
        for participant in self.data:
            for i in self.data[participant].order:
                counts[i] = 1 + counts.get(i, 0)
        return counts

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
