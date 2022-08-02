import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from collections import namedtuple

SYSTHREADS = min(4, cpu_count())
histogram = namedtuple('histogram', ['bins', 'counts'])


def compute_histogram(value, data, bins, nthreads=SYSTHREADS):
    def helper(quad):
        hist_quads = {}
        quad_df = data[data['QUADID'] == quad]
        for ch in range(32):
            adcs = quad_df[(quad_df['CHN'] == ch)][value]
            ys, _ = np.histogram(adcs, bins=bins)
            hist_quads[ch] = ys
        return quad, hist_quads

    results = Parallel(n_jobs=nthreads)(delayed(helper)(quad)
                                        for quad in 'ABCD')
    counts = {key: value for key, value in results}
    return histogram(bins, counts)


def move_mean(arr, n):
    return pd.Series(arr).rolling(n, center=True).mean().to_numpy()
