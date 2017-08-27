"""Clean, select and parse data.

Usage:
  ./data_cleaner.py -i <file> -o <file>
  ./data_cleaner.py -i <file> -s <n> -p <n> -o <file>

Options:
  -i <file>                Input data file
  -o <file>                Output data file or dir in the case of splitting
  -s <n>                   Number of instances to generate from data
  -p <n>                   percentage of overlapping between the instances

  --help                   show this screen
"""
from docopt import docopt
from scipy.stats import zscore
from pandas import read_csv
from constants import USECOLS
import numpy as np

USECOLS_STRING = ",".join(USECOLS)


def overlap(a, b, p):
    split_point = int(p * len(a))
    return np.concatenate([a, b[:split_point]])


def splitter(df, n_splits, p=0.25):
    n_cols = df.shape[0]
    ratio = int(n_cols / n_splits)
    spliting_indexes = [i for i in range(ratio, n_cols - ratio, ratio)]

    dfs = np.split(df, spliting_indexes, axis=0)
    for i in range(len(dfs) - 1):
        dfs[i] = overlap(dfs[i], dfs[i+1], p)
    return dfs


if __name__ == '__main__':
    opts = docopt(__doc__)

    infilename = opts['-i']
    outfilename = opts['-o']

    df = read_csv(infilename, sep=',', usecols=USECOLS)
    df = df.dropna()

    result = df.apply(zscore)

    result['semana'] = np.arange(0, df.shape[0])
    result['abundancia_norm'] = df['abundancia_norm']

    if opts['-s'] is None:
        result.to_csv(outfilename, sep=',', index=False)
    else:
        nsplit = int(opts['-s'])
        p = float(opts['-p'])

        results = splitter(result, nsplit, p)

        infilename_no_path = infilename.split('/')[-1].split('.')[0]
        for i, r in enumerate(results):
            out = outfilename + infilename_no_path + str(i) + '.csv'
            fmt = ",".join(["%s"] * df.shape[1])
            np.savetxt(out, r, fmt=fmt, delimiter=',', header=USECOLS_STRING)
