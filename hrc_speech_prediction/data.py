import json


# Note on terminology:
# - one session is the full recording of a participant
# - each session consists in several trials each corresponding to
#   an instruction (sheet)


class TrainData(object):
    """Contains training data.

    data:
    {paricipant_id: [(trial_instruction, trial_pairs)]
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
            for i, _ in self.data[participant]:
                counts[i] = 1 + counts.get(i, 0)
        return counts

    def dump(self, path):
        with open(path, 'w') as f:
            json.dump(self.data, f, indent=2)

    @classmethod
    def load(cls, path):
        with open(path) as f:
            return cls(json.load(f))
