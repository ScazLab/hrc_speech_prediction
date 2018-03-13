from unittest import TestCase

from hrc_speech_prediction.data import ALL_ACTIONS
from hrc_speech_prediction.context_model import ContextTreeModel, Node

import copy

class TestNode(TestCase):
    def setUp(self):
        self.empty_context = []
        self.context = ['a','b','c']
        self.contexts = [['a','b','c'],['d','e','f'],['g','h','i'],['a','g','e']]

    def test_empty_has_zero_n_children(self):
        x = Node()
        self.assertEqual(x.n_children, 0)

    def test_nonempty_has_n_children(self):
        x = Node()
        x.get_or_add_node(self.empty_context)
        self.assertEqual(x.n_children, 0)
      
        x.get_or_add_node(self.context)
        self.assertEqual(x.n_children, 1)

        x.get_or_add_node(self.context)
        self.assertEqual(x.n_children, 1)

        x.get_or_add_node(['b','c'])
        self.assertEqual(x.n_children, 2)

        self.assertEqual(x.n_children, len(x._children))

    def test_add_new_node_adds_child(self):
        x = Node()
        x.get_or_add_node(self.context)

        self.assertIn(self.context[0], x.seen_children)
        self.assertIn(self.context[1], x.get_or_add_node(self.context[:1]).seen_children)
        self.assertIn(self.context[2], x.get_or_add_node(self.context[:2]).seen_children)

    def test_add_branch_increments_count(self):
        x = Node()
        self.assertEqual(x._count, 0)
        branch = x.add_branch(self.context)
        self.assertEqual(branch._count, 1)
        branch = x.add_branch(self.context)
        self.assertEqual(branch._count, 2)
        self.assertEqual(x._children['a']._children['b']._count, 0)
        branch = x.add_branch(['a','b'])
        self.assertEqual(branch._count, 1)

class TestContextTreeModel(TestCase):
    def setUp(self):
        self.model = ContextTreeModel(ALL_ACTIONS)
        self.model.fit([[], [], ["a"], ["z", "b", "c"]], ["a", "a", "b", "d"])

    def test_n_children(self):
        self.assertEqual(self.model.root.n_children, 2)

    def test_fit_raises_exception_on_context_action_mismatch(self):
        self.assertRaises(ValueError, self.model.fit, [[], ['b'], ['c']], ['a','b'])

    def test_fit_empty_lists_does_nothing(self):
        self.old_modelroot = copy.deepcopy(self.model.root)
        self.assertEqual(self.old_modelroot, self.model.root)

        self.model.fit([], [])
        self.assertEqual(self.model.root.n_children, self.old_modelroot.n_children)
        #will be replaced by assertEqual(old_modelroot, self.model.root)
        self.assertEqual(self.old_modelroot, self.model.root)
        # for i in range(len(self.model.root._children.keys())):
        #     key = self.model.root._children.keys()[i]
        #     self.assertEqual(self.model.root._children[key]._count, self.old_modelroot._children[key]._count)
   
    def test_fit_one(self):
        fit_one = ContextTreeModel(ALL_ACTIONS)
        fit_one.fit([["a"]], ["b"])
        self.assertEqual(fit_one.root.seen_children, ["a"])
        self.assertTrue("b" in fit_one.root._children["a"].seen_children)


    def test_fit_several(self):
        self.assertIn("a", self.model.root.seen_children)
        self.assertIn("z", self.model.root.seen_children)

        self.assertIn("b", self.model.root._children["a"].seen_children)
        self.assertIn("b", self.model.root._children["z"].seen_children)
        self.assertIn("d", self.model.root._children["z"]._children["b"]._children["c"].seen_children)
        
        self.assertFalse("c" in self.model.root.seen_children)
        self.assertEqual(2, self.model.root._children["a"]._count)
        self.assertEqual(1, self.model.root._children["a"]._children["b"]._count)

    def test_total_of_counts_incremented_by_one_after_fit_one(self):
        fit_one = ContextTreeModel(ALL_ACTIONS)
        old_sum = fit_one.root.sum_children_counts
        fit_one.fit([["a"]], ["b"])
        self.assertEqual(fit_one.root.sum_children_counts + 1, old_sum)

    def test_predict_returns_array_of_correct_size(self):
        self.assertEqual(len(self.model.actions), len(self.model.predict(["a"])))
        self.assertEqual(len(self.model.actions), len(self.model.predict(["b"])))

    def test_empty_predicts_uniform(self):
        empty = ContextTreeModel(ALL_ACTIONS)
        prediction = empty.predict(["A"])
        for i in range(len(prediction)-1):
            self.assertEqual(prediction[i], prediction[i+1])

    def test_equals(self):
        # empty = ContextTreeModel(ALL_ACTIONS)
        # empty2 = ContextTreeModel(ALL_ACTIONS)
        # self.assertEqual(empty, empty2)
        empty = Node()
        empty2 = Node()
        self.assertEqual(empty, empty2)
