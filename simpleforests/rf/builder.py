from __future__ import division
import numpy as np

from .tree import InternalNode
from .tree import LeafNode

class TreeBuilder(object):

    def _find_split_parameters(self, X, Y, n_min_leaf, n_trials):
        """
        Compute parameters of the best split for the data X, Y.

        X: features, one data point per row
        Y: labels, one data point per row
        n_trials: the number of split dimensions to try.
        n_min_leaf: the minimum leaf size -- don't create a split with
            children smaller than this.

        Returns the pair (split_dim, split_threshold) or None if no appropriate
        split is found.  split_dim is an integer and split_threshold is a real
        number.

        Call self._information_gain(Y, Y_left, Y_right) to compute the
        information gain of a split.
        """
        n, d = X.shape
        best_ig = -1
        split_dim = None
        split_threshold = None
        for dim in np.random.randint(d, size=n_trials):
            sorted_X_d = np.sort(X[:, dim])
            thresholds = (sorted_X_d[0:n-1] + sorted_X_d[1:]) / 2.0
            for t in thresholds:
                idx = X[:, dim] < t
                Y_left = Y[idx]
                Y_right = Y[np.invert(idx)]
                if len(Y_left) < n_min_leaf or len(Y_right) < n_min_leaf:
                    continue
                ig = self._information_gain(Y, Y_left, Y_right)
                if ig > best_ig:
                    best_ig = ig
                    split_dim = dim
                    split_threshold = t
        if split_dim is not None:
            return split_dim, split_threshold
        return None


    def fit(self, X, Y, max_depth, n_min_leaf, n_trials):
        yhat = Y.mean(axis=0).reshape((1,-1))

        # short circuit for pure leafs
        if np.all(Y == Y[0]):
            return LeafNode(yhat)

        # avoid growing trees that are too deep
        if max_depth <= 0:
            return LeafNode(yhat)

        split_params = self._find_split_parameters(
                X, Y, n_min_leaf=n_min_leaf, n_trials=n_trials)

        # if we didn't find a good split point then become leaf
        if split_params is None:
            return LeafNode(yhat)

        split_dim, split_threshold = split_params

        mask_l = X[:,split_dim] < split_threshold
        mask_r = np.logical_not(mask_l)

        # refuse to make leafs that are too small
        if np.sum(mask_l) < n_min_leaf or \
                np.sum(mask_r) < n_min_leaf:
            raise Exception("Leaf too small")

        # otherwise split this node recursively
        left_child = self.fit(
                X[mask_l],
                Y[mask_l],
                max_depth=max_depth - 1,
                n_min_leaf=n_min_leaf,
                n_trials=n_trials)

        right_child = self.fit(
                X[mask_r],
                Y[mask_r],
                max_depth=max_depth - 1,
                n_min_leaf=n_min_leaf,
                n_trials=n_trials)

        return InternalNode(
                dim=split_dim,
                threshold=split_threshold,
                left_child=left_child,
                right_child=right_child)


class ClassificationTreeBuilder(TreeBuilder):

    def _entropy(self, x):
        x = x[x>0]
        return -np.sum(x*np.log(x))

    def _information_gain(self, y, y_l, y_r):
        n = y.shape[0]
        n_l = y_l.shape[0]
        n_r = y_r.shape[0]

        H = self._entropy(y.mean(axis=0))
        H_l = self._entropy(y_l.mean(axis=0))
        H_r = self._entropy(y_r.mean(axis=0))

        return H - n_l/n * H_l - n_r/n * H_r


class RegressionTreeBuilder(TreeBuilder):

    def _information_gain(self, y, y_l, y_r):
        assert y.size == y_l.size + y_r.size
        assert y_l.size > 0
        assert y_r.size > 0

        sse = np.sum((y - y.mean())**2)
        sse_l = np.sum((y_l - y_l.mean())**2)
        sse_r = np.sum((y_r - y_r.mean())**2)

        return sse - sse_l - sse_r

