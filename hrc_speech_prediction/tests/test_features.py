from unittest import TestCase
from collections import OrderedDict

from hrc_speech_prediction.data import Trial, Session, TrainData, TimedAction
from hrc_speech_prediction.features import get_context_features

import numpy as np


class TestContextFeatures(TestCase):

    def setUp(self):
        self.pairs1 = [(TimedAction('a', .1, .2), []),
                       (TimedAction('b', .3, .4), []),
                       (TimedAction('c', .3, .4), []),
                       ]
        self.pairs2 = [(TimedAction('c', 1.6, 1.8), []),
                       (TimedAction('b', .3, .4), []),
                       (TimedAction('a', .3, .4), []),
                       ]
        self.trial1 = Trial('B', self.pairs1, .0)
        self.trial2 = Trial('A', self.pairs2, 1.)
        self.td = TrainData(OrderedDict([
            ('P1', Session([self.trial1])),
            ('P2', Session([self.trial2]))
        ]))

    def test_get_context_features(self):
        X, labels = get_context_features(self.td)
        self.assertEqual(set(labels), set('abc'))
        a = labels.index('a')
        b = labels.index('b')
        c = labels.index('c')
        np.testing.assert_equal(
            X[:, [a, b, c]],
            [[1, 1, 1],
             [0, 1, 1],
             [0, 0, 1],
             [1, 1, 1],
             [1, 1, 0],
             [1, 0, 0]]
        )
