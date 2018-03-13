import numpy as np


class ContextTreeModel(object):
    def __init__(self, actions, eps=0.15):
        "docstring"

        self.actions = actions
        self.eps = eps
        self.root = Node()

    def curr(self, cntxt):
        return self.root.get_or_add_node(cntxt)

    def fit(self, ctxts, acts):
        if len(ctxts) != len(acts):
            raise ValueError("Context and action sizes not matching.")
        for c, a in zip(ctxts, acts):
            self.root.add_branch(c + [a])

    def predict(self, ctxt):
        curr = self.root.get_or_add_node(ctxt)
        probs = curr._get_context_probs(self.eps, self.actions)

        return self._apply_eps(probs)

    def _apply_eps(self, p):
        u = np.array([1.0 / len(self.actions) for i in self.actions])

        return (1.0 - self.eps) * p + (u * self.eps)

    def __str__(self):
        return self.root.__str__()


class Node(object):
    def __init__(self):
        "A state in a task trajectory"
        self._count = 0.0
        self._children = {}  # keys: action taken, val: list of nodes

    @property
    def n_children(self):
        return len(self._children)

    @property
    def seen_children(self):
        return self._children.keys()

    @property
    def sum_children_counts(self):
        return sum([self._children[c]._count for c in self.seen_children])

    def _increment_count(self):
        self._count += 1.0
        return self

    def __eq__(self, other):
        if self.n_children == 0 and other.n_children == 0:
            return True
        # test for same number of children
        if self.n_children == other.n_children:
            for key in self._children:
                # test for same keys
                if key not in other.seen_children:
                    return False
                # test for same count
                if self._children[key]._count != other._children[key]._count:
                    return False
                # test for same children, can't be == because not defined yet
                if not self._children[key].__eq__(other._children[key]):
                    return False

        else:
            return False
        return True

    def get_or_add_node(self, cntxt):
        if not cntxt:
            return self

        c = cntxt[0]

        if c not in self.seen_children:
            self._children[c] = Node()

        return self._children[c].get_or_add_node(cntxt[1:])

    def add_branch(self, cntxt):
        c = self.get_or_add_node(cntxt)
        c._increment_count()

        return c

    def _get_context_probs(self, eps, actions):
        "Returns probabilities for taking each child action based \
        only on how many times each child has been visited"

        s = 1.0 / (self.sum_children_counts + .00001)

        return np.array([
            self._children[k]._count * s if k in self.seen_children else 0.0
            for k in actions
        ])

    def __str__(self, level=0, val="init"):
        ret = "\t" * level + "{}: {}\n".format(val, self._count)
        for k, v in self._children.items():
            ret += v.__str__(level + 1, k)
        return ret
