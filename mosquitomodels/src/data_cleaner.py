"""Clean, select and parse data.

Usage:
  ./data_cleaner.py -i <file> -o <file>
  ./data_cleaner.py -i <file> -s <n> -o <file>

Options:
  -i <file>                Input data file
  -o <file>                Output data file or dir in the case of splitting
  -s <file>                Split data to obtain instances for irace, give dir

  --help                   show this screen
"""
from docopt import docopt
from scipy.stats import zscore
from pandas import read_csv
import numpy as np

usecols = ['semana',
            'fecha',
            'abundancia',
            'abundancia_norm',
            'ndvirural',
            'ndvirurallag1',
            'ndviurban',
            'ndviurbanlag1',
            'ndwirural',
            'ndwirurallag1',
            'ndwiurban',
            'ndwiurbanlag1',
            'lstdrural',
            'lstdrurallag3',
            'lstdurban',
            'lstdurbanlag3',
            'lstnrural',
            'lstnrurallag1',
            'lstnurban',
            'lstnurbanlag2',
            'trmmrural',
            'trmmrurallag3',
            'dias_frio_rural5',
            'grados_frio_rural5',
            'dias_frio_rural10',
            'grados_frio_rural10', ]

usecols = ['semana',
           'abundancia_norm',
           'ndvirurallag1',
           'ndwirurallag1',
           'lstdrurallag3',
           'lstnrurallag1',
           'trmmrurallag3',
           'dias_frio_rural10',
           'grados_frio_rural10', ]

usecols = usecols[:-2]  # Ignore last two

usecols_string = ",".join(usecols)


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
    split = int(opts['-s'])

    df = read_csv(infilename, sep=',', usecols=usecols)
    df = df.dropna()

    result = df.apply(zscore)

    result['semana'] = np.arange(0, df.shape[0])
    result['abundancia_norm'] = df['abundancia_norm']

    if split is None:
        result.to_csv(outfilename, sep=',', index=False)
        result = read_csv(outfilename, sep=',')  # Debug
        print(result)  # Debug
    else:
        results = splitter(result, split)

        for i, r in enumerate(results):
            out = outfilename + str(i) + '.csv'
            fmt = ",".join(["%s"] * df.shape[1])
            np.savetxt(out, r, fmt=fmt, delimiter=',', header=usecols_string)