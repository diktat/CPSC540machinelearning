import numpy as np
import matplotlib.pyplot as plt

import rf
from tools import *

def load_data():
    n_per = 200

    X_1 = np.random.standard_normal(size=(n_per, 2)) + np.array([2, 1])
    X_2 = np.random.standard_normal(size=(n_per, 2)) + np.array([0.5, -2])
    X_3 = np.random.standard_normal(size=(n_per, 2)) + np.array([-2, 0.5])
    Y_1 = np.zeros(n_per)
    Y_2 = np.ones(n_per)
    Y_3 = np.ones(n_per) * 2
    X = np.concatenate([X_1, X_2, X_3], axis=0)
    Y = np.concatenate([Y_1, Y_2, Y_3], axis=0)

    return X, Y

def plot_results(X, Y, Yhat, forest):
    grid_extend = [X[:,0].min(), X[:,0].max(), X[:,1].min(), X[:,1].max()]
    Ux, Uy = np.meshgrid(
            np.linspace(grid_extend[0], grid_extend[1]),
            np.linspace(grid_extend[2], grid_extend[3]),
        )

    X_grid = np.concatenate([
        Ux.reshape((-1,1)), Uy.reshape((-1,1))],
        axis=1)

    Y_probs, Y_probs_per_tree = forest.predict(X_grid, return_tree_predictions=True)
    Y_hat = max_of_n_prediction(Y_probs)

    plt.figure()
    colors = np.array([[1,0,0], [0,1,0], [0,0,1]])
    img = image_from_predictions(Y_hat, max_prob_of_n_prediction(Y_probs), colors, Ux.shape)
    plt.imshow(img, extent=grid_extend, origin='lower')
    plt.scatter(X[:,0], X[:,1], c=colors_from_predictions(Y, colors))
    plt.savefig("synthetic_classification.png")

    plt.figure()
    for i in xrange(25):
        plt.subplot(5,5,i)
        Y_hat = max_of_n_prediction(Y_probs_per_tree[i])
        img = image_from_predictions(Y_hat, max_prob_of_n_prediction(Y_probs_per_tree[i]), colors, Ux.shape)
        plt.imshow(img, extent=grid_extend, origin='lower')

    plt.savefig("synthetic_classification_tree_predictions.png")

def run_test():
    X, Y = load_data()

    forest = rf.ClassificationForest().fit(
            X,
            encode_one_of_n(Y),
            n_trees=25,
            max_depth=3,
            n_min_leaf=3,
            n_trials=1)
    Yhat = forest.predict(X)
    Yhat = max_of_n_prediction(Yhat)

    plot_results(X, Y, Yhat, forest)

    acc = np.mean(Y == Yhat)
    print acc
    assert acc > 0.9

if __name__ == "__main__":
    run_test()

