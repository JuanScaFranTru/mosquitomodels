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
from sklearn.tree import DecisionTreeRegressor


def add_pca_to_model(model):
    class PCAModel(model):
        pca = True
    return PCAModel


def MLPRegressorProxy(alpha, neurons, layers):
    hidden_layer_sizes = tuple([neurons] * layers)
    return MLPRegressor(hidden_layer_sizes=hidden_layer_sizes, solver='lbfgs',
                        activation='logistic', alpha=alpha)


MODELS = {
    'dtr': DecisionTreeRegressor,
    'knnr': KNeighborsRegressor,
    'mlpr': MLPRegressorProxy,
    'svr': SVR,
    'pcaknnr': add_pca_to_model(KNeighborsRegressor),
    'pcadtr': add_pca_to_model(DecisionTreeRegressor),
    'linear': LinearRegression,
    "ridge": RidgeCV,
}
