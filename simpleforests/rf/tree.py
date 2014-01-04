import numpy as np

class LeafNode(object):
    def __init__(self, yhat):
        self.yhat = yhat

    def predict(self, X):
        return np.outer(np.ones(X.shape[0]), self.yhat)


class InternalNode(object):
    def __init__(self, dim, threshold, left_child, right_child):
        self.dim = dim
        self.threshold = threshold
        self.left_child = left_child
        self.right_child = right_child

        assert left_child is not None
        assert right_child is not None

    def predict(self, X):
        # branch masks
        left_mask = X[:,self.dim] <= self.threshold
        right_mask = np.logical_not(left_mask)

        # sanity check: everybody needs to go somewhere and nobody can
        # go both directions
        assert np.all(np.logical_xor(left_mask, right_mask))

        go_left = np.any(left_mask)
        go_right = np.any(right_mask)

        # evaluate branches
        if go_left:
            Y_left = self.left_child.predict(X[left_mask])

        if go_right:
            Y_right = self.right_child.predict(X[right_mask])

        if go_left and go_right:
            # sanity check: make sure we got compatible results from the two subtrees
            assert Y_left.dtype == Y_right.dtype
            assert Y_left.shape[1] == Y_right.shape[1]

            # rebuild Y in the same order as X
            Y = np.zeros((X.shape[0], Y_left.shape[1]), dtype=Y_left.dtype)
            Y[left_mask,:] = Y_left
            Y[right_mask,:] = Y_right

            return Y

        elif go_left:
            return Y_left

        elif go_right:
            return Y_right

        else:
            raise Exception("This can't happen.")

