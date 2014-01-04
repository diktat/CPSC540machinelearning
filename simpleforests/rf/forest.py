import numpy as np

from .builder import RegressionTreeBuilder
from .builder import ClassificationTreeBuilder


def _bootstrap(X, Y):
    sample = np.random.randint(X.shape[0], size=X.shape[0])
    return X[sample], Y[sample]


class Forest(object):

    def predict(self, X, return_tree_predictions=False):
        Ys = [ tree.predict(X) for tree in self.trees ]
        Y = 1./len(Ys) * sum(Ys)
        if return_tree_predictions:
            return Y, Ys
        else:
            return Y

    def fit(self,
            X, Y,
            n_trees,
            **kwargs):

        self.trees = [
            self._tree_builder().fit(
                *_bootstrap(X, Y),
                **kwargs)
            for _ in xrange(n_trees)
            ]

        return self


class RegressionForest(Forest):
    def _tree_builder(self):
        return RegressionTreeBuilder()

    def fit(self,
            X, Y,
            n_trees,
            **kwargs):

        assert X.ndim == 2
        assert Y.ndim == 2
        assert Y.shape[1] == 1

        return super(RegressionForest, self).fit(X, Y, n_trees, **kwargs)

class ClassificationForest(Forest):
    def _tree_builder(self):
        return ClassificationTreeBuilder()

    def fit(self,
            X, Y,
            n_trees,
            **kwargs):

        assert X.ndim == 2
        assert Y.ndim == 2

        return super(ClassificationForest, self).fit(X, Y, n_trees, **kwargs)
