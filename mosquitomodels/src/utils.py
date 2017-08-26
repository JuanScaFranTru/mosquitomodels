import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import cross_val_score


def to_classes(xs, start=0, stop=1, nclasses=10):
    """Return a Classes array"""
    classified = xs
    intervals = np.linspace(start, stop, nclasses+1)
    for i in range(len(xs)):
        j = 0
        while not (xs[i] >= intervals[j] and xs[i] <= intervals[j + 1]):
            j += 1
            if j == len(intervals) - 1:
                break
        classified[i] = j
    assert len(xs) == len(classified)
    return classified


def smoothing(xs):
    n = len(xs)
    for i in range(1, n - 1):
        xs[i] = sum([xs[j] for j in [i - 1, i, i + 1]]) / 3
    xs[0] = xs[1]
    xs[n - 1] = xs[n - 2]

    return xs


def save_data(filename, weeks, y_true, y_pred):
    """Save the predictions to a csv."""
    assert len(weeks) == len(y_true)
    assert len(y_pred) == len(y_true)
    X = np.transpose(np.array([weeks, y_true, y_pred]))
    np.savetxt(filename, X, delimiter=',')


def load_data(filename='data/all_data.csv'):
    """Load the training data set."""

    data = np.loadtxt(filename, delimiter=',', skiprows=1)
    n_rows, n_cols = data.shape
    assert n_cols > 3

    weeks = data[:, 0]
    ts = smoothing(data[:, 1])
    xs = data[:, 2:]

    return n_cols, weeks, ts, xs


def stats(xs, ts, model, n_splits=5):
    cv = TimeSeriesSplit(n_splits)
    scores = cross_val_score(model, xs, ts, cv=cv.split(xs),
                             scoring='neg_mean_squared_error')
    scores = np.sqrt(-scores)
    mean = np.mean(scores)
    std_dev = np.std(scores)

    return scores, mean, std_dev


def print_stats(scores, mean, std_dev, title='Stats'):
    print()
    print(title)
    print('-' * len(title))
    print('')
    print('Model Scores: ', scores)
    print('Mean Score: ', mean)
    print('Standard Deviation of Score: ', std_dev)


def save_plot(filename, weeks, y_true, y_pred, xlabel='Weeks',
              ylabel='Normalized abundance', label_true='Ground truth',
              label_pred='Model', dpi=300):

    plt.plot(weeks, y_true, color='c', linestyle='dashed', label=label_true)
    plt.plot(weeks, y_pred, color='g', linestyle='solid', label=label_pred)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    plt.legend()

    plt.savefig(filename, format='eps', dpi=dpi)
