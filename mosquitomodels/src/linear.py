"""Train a model

Usage:
  ./linear.py -i <file> -m <model> [--n_comp <n>]
  ./linear.py -h | --help

Options:
  -i <file>             Input file
  -m <model>             Model to be selected

  --n_comp <n>             PCA_model parameter

  --help                   show this screen
"""
from docopt import docopt
from utils import load_data, stats
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import LinearRegression
import numpy as np


class PositiveRidgeCV(RidgeCV):
    def predict(self, X):
        y = super(PositiveRidgeCV, self).predict(X)
        y = np.maximum([0] * len(y), y)
        return y


class PositiveLinearRegression(LinearRegression):
    def predict(self, X):
        y = super(LinearRegression, self).predict(X)
        y = np.maximum([0] * len(y), y)
        return y


models = {
    'linear': PositiveLinearRegression,
    'ridge': PositiveRidgeCV,
}


if __name__ == '__main__':
    opts = docopt(__doc__)
    filename = opts['-i']
    data_len, weeks, ts, xs = load_data(filename=filename)

    model = models[opts['-m']]
    linear_model = model()

    # Models uses PCA?
    if opts['--n_comp'] is not None:
        n_components = int(opts['--n_comp'])
        pca = PCA(n_components=n_components)
        xs = pca.fit(xs).transform(xs)

    linear_model.fit(xs, ts)

    scores, mean, std = stats(xs, ts, linear_model)
    prediction = linear_model.predict(xs)
