from unittest import TestCase

from hrc_speech_prediction.data import Trial, Session


class TestTrial(TestCase):

    def setUp(self):
        self.pairs = [(('action_1', .1, .2),
                       [('blah blah', .0, .05), ('blah again', .06, .07)]),
                      (('action_2', .3, .4), [('do action2', .25, .3)]),
                      (('action_3', .6, .8), [('nonsense', .25, .3)]),
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
        self.pairs1 = [(('action_1', .1, .2),
                        [('blah blah', .0, .05), ('blah again', .06, .07)]),
                       (('action_2', .3, .4), [('do action2', .25, .3)]),
                       ]
        self.pairs2 = [(('action_3', 1.6, 1.8), [('nonsense', 1.25, 1.3)])]
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
