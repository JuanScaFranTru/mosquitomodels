"""Plot input data (not predictions) of different cities.

Usage:
  ./plotdata.py [--skipcols <n>] [--nolag] <file> ...

Options:
  --skipcols <n>           plot the data skipping the first <n> cols
  --nolag                  only plot variables with no lag
  --help                   show this screen
"""
from docopt import docopt
from pandas import read_csv
from scipy.stats import zscore
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from utils import get_filename_from_path
from constants import INDEPENDENT_NO_LAG, DEPENDENT, XTICKS_HEATMAP
import numpy as np
import seaborn as sns


if __name__ == '__main__':
    sns.set()
    opts = docopt(__doc__)
    infilenames = opts['<file>']
    nolag = opts['--nolag']

    for i, fn in enumerate(infilenames):
        df = read_csv(fn, sep=',')
        possible_cols = list(df.columns.values)
        if opts['--skipcols'] is not None:
            skipcols = int(opts['--skipcols'])
            dropcols = [x for x in range(skipcols)]
            df = df.drop(df.columns[dropcols], axis=1)
        if nolag:
            dropcols = [c for c in possible_cols if 'lag' in c]
            df = df.drop(dropcols, axis=1)

        df = df.dropna()
        df = df.apply(zscore)

        f = plt.figure(i)

        step = int(df.shape[0] / XTICKS_HEATMAP)
        xticklabels = np.arange(0, df.shape[0], step)

        ax = sns.heatmap(np.transpose(df), cmap='YlGnBu',
                         xticklabels=xticklabels)

        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

        f.suptitle(get_filename_from_path(fn))
        plt.yticks(rotation='horizontal')

    plt.show()
