from unittest import TestCase

from hrc_speech_prediction.data import ALL_ACTIONS
from hrc_speech_prediction.context_model import ContextTreeModel, Node


class TestNode(TestCase):

    def setUp(self):
        pass

    def test_empty_has_zero_n_children(self):
        raise NotImplementedError

    def test_nonempty_has_n_children(self):
        raise NotImplementedError

    def test_add_new_node_adds_child(self):
        raise NotImplementedError

    def test_get_or_add_node_increments_count(self):
        raise NotImplementedError


class TestContextTreeModel(TestCase):
    def setUp(self):
        self.model = ContextTreeModel(ALL_ACTIONS)
        self.model.fit([[], [], ["a"], ["z", "b", "c"]], ["a", "a", "b", "d"])

    def test_n_children(self):
        self.assertEqual(self.model.root.n_children, 2)

    def test_fit_raises_exception_on_context_action_mismatch(self):
        raise NotImplementedError

    def test_fit_empty_lists_does_nothing(self):
        raise NotImplementedError

    def test_fit_one(self):
        raise NotImplementedError

    def test_fit_several(self):
        raise NotImplementedError

    def test_total_of_counts_incremented_by_one_after_fit_one(self):
        raise NotImplementedError

    def test_predict_returns_array_of_correct_size(self):
        raise NotImplementedError

    def test_empty_predicts_uniform(self):
        raise NotImplementedError
