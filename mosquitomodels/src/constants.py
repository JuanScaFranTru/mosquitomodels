USECOLS = [
    'Week',
    'Normalized abundancy',
    'Country NDWI lag 1',
    'Country NDVI lag 1',
    'Country LST day lag 3',
    'Country LST night lag 1',
    'Country TRMM lag 3',
    'Country cold days 10',
    'Country cold degrees 10',
]
"""Colums that were selected to be used in the models."""


FIRSTCOLS = [
    'Week',
    'Date',
    'Normalized abundancy',
]

DEPENDENT = [
    'Abundancy',
]

INDEPENDENT = [
    'Country LST day',
    'Country LST night',

    'Country NDVI',
    'Country NDWI',
    'Country TRMM',

    'Country cold days 10',
    'Country cold days 5',

    'Country cold degrees 10',
    'Country cold degrees 5',

    'Urban LST day',
    'Urban LST night',

    'Urban NDVI',
    'Urban NDWI',

    'Country LST day lag 3',
    'Country LST night lag 1',

    'Country NDVI lag 1',
    'Country NDWI lag 1',
    'Country TRMM lag 3',

    'Urban LST day lag 3',
    'Urban LST night lag 2',

    'Urban NDVI lag 1',
    'Urban NDWI lag 1',
]

INDEPENDENT_NO_LAG = [c for c in INDEPENDENT if 'lag' not in c]

ALLCOLS = FIRSTCOLS + DEPENDENT + INDEPENDENT

XTICKS_HEATMAP = 10
"""Number of xticklabels in heatmaps."""
