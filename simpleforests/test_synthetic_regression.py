import numpy as np
import matplotlib.pyplot as plt

import rf

def load_data():
    n_per = 50

    X_1, X_2 = np.meshgrid(
        np.linspace(-np.pi, np.pi, n_per),
        np.linspace(-np.pi, np.pi, n_per))

    #Y = np.sin(X_1 + X_2).reshape((-1,1))
    Y = np.exp(-0.5*(X_1**2 + X_2**2)).reshape((-1,1))
    X = np.concatenate([
        X_1.reshape((-1,1)),
        X_2.reshape((-1,1))],
        axis=1)
    Y += 0.25*np.random.standard_normal(size=Y.shape)

    return X, Y

def plot_results(X, Yhat, forest):
    grid_extend = [X[:,0].min(), X[:,0].max(), X[:,1].min(), X[:,1].max()]
    Ux, Uy = np.meshgrid(
            np.linspace(grid_extend[0], grid_extend[1]),
            np.linspace(grid_extend[2], grid_extend[3]),
        )

    X_grid = np.concatenate([
        Ux.reshape((-1,1)), Uy.reshape((-1,1))],
        axis=1)

    Y_probs, Y_probs_per_tree = forest.predict(X_grid, return_tree_predictions=True)

    plt.figure()
    colors = np.array([[1,0,0], [0,1,0], [0,0,1]])
    plt.imshow(Y_probs.reshape(Ux.shape), extent=grid_extend, origin='lower')
    plt.savefig("synthetic_regression.png")

    plt.figure()
    for i in xrange(25):
        plt.subplot(5,5,i)
        Y_hat = Y_probs_per_tree[i]
        plt.imshow(Y_probs_per_tree[i].reshape(Ux.shape), extent=grid_extend, origin='lower')

    plt.savefig("synthetic_regression_tree_predictions.png")

def run_test():
    X, Y = load_data()

    forest = rf.RegressionForest().fit(
            X, Y,
            n_trees=25,
            max_depth=7,
            n_min_leaf=1,
            n_trials=1)
    Yhat = forest.predict(X)

    plot_results(X, Yhat, forest)

    mse = np.mean((Y-Yhat)**2)
    print mse
    assert mse < 0.06

if __name__ == "__main__":
    run_test()
