from unittest import TestCase
from collections import OrderedDict

from hrc_speech_prediction.data import (Trial, Session, TrainData, TimedAction,
                                        TimedUtterance)


class TestTrial(TestCase):

    def setUp(self):
        self.pairs = [(TimedAction('action_1', .1, .2),
                       [TimedUtterance('blah blah', .0, .05),
                        TimedUtterance('blah again', .06, .07)]),
                      (TimedAction('action_2', .3, .4),
                       [TimedUtterance('do action2', .25, .3)]),
                      (TimedAction('action_3', .6, .8),
                       [TimedUtterance('nonsense', .25, .3)]),
                      ]
        self.trial = Trial('A', self.pairs, .0)

    def test_n_samples(self):
        self.assertEqual(self.trial.n_samples, 3)

    def test_utterances(self):
        utterances = ['blah blah blah again', 'do action2', 'nonsense']
        self.assertEqual(self.trial.utterances, utterances)

    def test_labels(self):
        self.assertEqual(self.trial.labels,
                         ['action_1', 'action_2', 'action_3'])

    def test_first_id_not_set(self):
        with self.assertRaises(ValueError):
            self.trial.first_id

    def test_first_id(self):
        self.trial.set_first_id(5)
        self.assertEqual(self.trial.first_id, 5)

    def test_ids(self):
        self.trial.set_first_id(5)
        self.assertEqual(self.trial.ids, [5, 6, 7])

    def test_get_pair_from_id(self):
        self.trial.set_first_id(5)
        self.assertEqual(self.trial.get_pair_from_id(6), self.pairs[1])


class TestSession(TestCase):

    def setUp(self):
        self.pairs1 = [(TimedAction('action_1', .1, .2),
                        [TimedUtterance('blah blah', .0, .05),
                         TimedUtterance('blah again', .06, .07)]),
                       (TimedAction('action_2', .3, .4),
                        [TimedUtterance('do action2', .25, .3)]),
                       ]
        self.pairs2 = [(TimedAction('action_3', 1.6, 1.8),
                        [TimedUtterance('nonsense', 1.25, 1.3)])]
        self.session = Session([Trial('B', self.pairs1, .0),
                                Trial('A', self.pairs2, 1.)])

    def test_n_samples(self):
        self.assertEqual(self.session.n_samples, 3)

    def test_utterances(self):
        utterances = ['blah blah blah again', 'do action2', 'nonsense']
        self.assertEqual(list(self.session.utterances), utterances)

    def test_labels(self):
        self.assertEqual(list(self.session.labels),
                         ['action_1', 'action_2', 'action_3'])

    def test_first_id_not_set(self):
        with self.assertRaises(ValueError):
            self.session.first_id

    def test_first_id(self):
        self.session.set_first_id(5)
        self.assertEqual(self.session.first_id, 5)

    def test_ids(self):
        self.session.set_first_id(5)
        self.assertEqual(list(self.session.ids), [5, 6, 7])

    def test_get_pair_from_id(self):
        self.session.set_first_id(5)
        self.assertEqual(self.session.get_pair_from_id(6), self.pairs1[1])
        self.assertEqual(self.session.get_pair_from_id(7), self.pairs2[0])


class TestTrainData(TestCase):

    def setUp(self):
        self.pairs1 = [(TimedAction('action_1', .1, .2),
                        [TimedUtterance('blah blah', .0, .05),
                         TimedUtterance('blah again', .06, .07)]),
                       (TimedAction('action_2', .3, .4),
                        [TimedUtterance('do action2', .25, .3)]),
                       ]
        self.pairs2 = [(TimedAction('action_3', 1.6, 1.8),
                        [TimedUtterance('nonsense', 1.25, 1.3)])]
        self.pairs3 = [(TimedAction('action_1', .2, .3),
                        [TimedUtterance('Hello robot', .1, .13)])]
        self.trial1 = Trial('B', self.pairs1, .0)
        self.trial2 = Trial('A', self.pairs2, 1.)
        self.trial3 = Trial('A', self.pairs3, 4.)
        self.td = TrainData(OrderedDict([
            ('P1', Session([self.trial1, self.trial2])),
            ('P2', Session([self.trial3]))
        ]))

    def test_participants(self):
        self.assertEqual(self.td.participants, ['P1', 'P2'])

    def test_n_participants(self):
        self.assertEqual(self.td.n_participants, 2)

    def test_n_samples(self):
        self.assertEqual(self.td.n_samples, 4)

    def test_ids(self):
        self.assertEqual(list(self.td.ids), list(range(self.td.n_samples)))

    def test_utterances(self):
        utterances = ['blah blah blah again', 'do action2', 'nonsense',
                      'Hello robot']
        self.assertEqual(list(self.td.utterances), utterances)

    def test_labels(self):
        labels = ['action_1', 'action_2', 'action_3', 'action_1']
        self.assertEqual(list(self.td.labels), labels)

    def test_get_pair_from_id(self):
        self.assertEqual(self.td.get_pair_from_id(1), self.pairs1[1])
        self.assertEqual(self.td.get_pair_from_id(2), self.pairs2[0])
        self.assertEqual(self.td.get_pair_from_id(3), self.pairs3[0])

    def test_all_trials(self):
        self.assertEqual(list(self.td.all_trials()),
                         [self.trial1, self.trial2, self.trial3])

    def test_count_by_instructions(self):
        self.assertEqual(self.td.count_by_instructions(), {'A': 2, 'B': 1})
