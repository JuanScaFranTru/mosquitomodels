"""Train a model

Usage:
  ./models.py -i <s> model <s> param <n> ...
  ./models.py -i <s> model <s> --predict <s> --sp <n> -p <s>
  ./models.py -i <s> model <s> --predict <s> --sp <n>
  ./models.py -h | --help

"""
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn.tree import DecisionTreeRegressor


def make_custom_model(model):
    class CustomModel(model):
        def predict(self, X):
            y = super(CustomModel, self).predict(X)
            y = np.maximum([0] * len(y), y)
            return y
    return CustomModel


def add_pca_to_model(model):
    class PCAModel(model):
        pca = True
    return PCAModel


mlpr = make_custom_model(MLPRegressor)


def MLPRegressorProxy(alpha, neurons, layers):
    hidden_layer_sizes = tuple([neurons] * layers)
    return mlpr(hidden_layer_sizes=hidden_layer_sizes, solver='lbfgs',
                activation='logistic', alpha=alpha)


MODELS = {
    'dtr': make_custom_model(DecisionTreeRegressor),
    'knnr': make_custom_model(KNeighborsRegressor),
    'mlpr': MLPRegressorProxy,
    'svr': make_custom_model(SVR),
    'pcaknnr': add_pca_to_model(make_custom_model(KNeighborsRegressor)),
    'linear': make_custom_model(LinearRegression),
    "ridge": make_custom_model(RidgeCV),
}
