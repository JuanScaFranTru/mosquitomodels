"""Train a model

Usage:
  ./linear.py -i <s> --model [--n_comp <n>]
  ./linear.py -h | --help

Options:
  -i       <s>             Input file
  --model  <s>             Model to be selected

  --n_comp <n>             PCA_model parameter

  --help                   show this screen
"""
from docopt import docopt
from utils import load_data, print_stats, save_data, stats
from sklearn.decomposition import PCA
from models.positive_models import PositiveLinearRegression
from models.positive_models import PositiveRidgeCV


models = {
    'linear': PositiveLinearRegression,
    'ridge': PositiveRidgeCV,
}


if __name__ == '__main__':
    opts = docopt(__doc__)
    filename = opts['-i']

    data_len, weeks, ts, xs = load_data(filename=filename)

    model = models[opts['--model']]
    linear_model = model()

    # Models uses PCA?
    if opts['--n_comp'] is not None:
        n_components = int(opts['--n_comp'])
        pca = PCA(n_components=n_components)
        xs_pca = pca.fit(xs).transform(xs)
        model.fit(xs_pca, ts)
        mean, std_dev = stats(xs_pca, ts, linear_model)
    else:
        model.fit(xs, ts)
        mean, std_dev = stats(xs, ts, model)

    print(mean)
