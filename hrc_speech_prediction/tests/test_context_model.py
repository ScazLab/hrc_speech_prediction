from unittest import TestCase

from hrc_speech_prediction.data import ALL_ACTIONS
from hrc_speech_prediction.context_model import ContextTreeModel, Node


class TestNode(TestCase):

    def setUp(self):
        self.empty_context = []
        self.context = ['a','b','c']
        self.contexts = [['a','b','c'],['d','e','f'],['g','h','i'],['a','g','e']]
        pass

    def test_empty_has_zero_n_children(self):
        self.x = Node()
        self.assertEqual(self.x.n_children, 0)
        #no deletion code
        # self.x.get_or_add_node(self.contxt)

    def test_nonempty_has_n_children(self):
        self.x = Node()
        self.x.get_or_add_node(self.empty_context)
        self.assertEqual(self.x.n_children, 0)
      
        self.x.get_or_add_node(self.context)
        self.assertEqual(self.x.n_children,1)

        self.x.get_or_add_node(self.context)
        self.assertEqual(self.x.n_children, 1)

        self.x.get_or_add_node(['b','c'])
        self.assertEqual(self.x.n_children, 2)

        self.assertEqual(self.x.n_children, len(self.x._children))

    def test_add_new_node_adds_child(self):
        self.x = Node()
        self.x.get_or_add_node(self.context)

        self.assertTrue(self.context[0] in self.x._children.keys())
        self.assertTrue(self.context[1] in self.x._children[self.context[0]]._children.keys())
        self.assertTrue(self.context[2] in self.x._children[self.context[0]]._children[self.context[1]]._children.keys())




    def test_add_branch_increments_count(self):
        self.x = Node()
        self.assertEqual(self.x._count,0)
        self.branch = self.x.add_branch(self.context)
        self.assertEqual(self.branch._count, 1)
        self.branch = self.x.add_branch(self.context)
        self.assertEqual(self.branch._count, 2)

        self.assertEqual(self.x._children['a']._children['b']._count, 0)
        self.branch = self.x.add_branch(['a','b'])
        self.assertEqual(self.branch._count, 1)

class TestContextTreeModel(TestCase):
    def setUp(self):
        self.model = ContextTreeModel(ALL_ACTIONS)
        self.model.fit([[], [], ["a"], ["z", "b", "c"]], ["a", "a", "b", "d"])

    # # def test_n_children(self):
    #     self.assertEqual(self.model.root.n_children, 2)

    # def test_fit_raises_exception_on_context_action_mismatch(self):
    #     raise NotImplementedError

    # def test_fit_empty_lists_does_nothing(self):
    #     raise NotImplementedError

    # def test_fit_one(self):
    #     raise NotImplementedError

    # def test_fit_several(self):
    #     raise NotImplementedError

    # def test_total_of_counts_incremented_by_one_after_fit_one(self):
    #     raise NotImplementedError

    # def test_predict_returns_array_of_correct_size(self):
    #     raise NotImplementedError

    # def test_empty_predicts_uniform(self):
    #     raise NotImplementedError