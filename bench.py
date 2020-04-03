import numpy as np
import pandas as pd
import rollingrank

from pythonbenchmark import measure

x = np.random.rand(1024 * 1024)
x_nan = x.copy()
x_nan[np.arange(0, x.size, 2)] = np.nan
x_small = np.random.rand(16 * 1024)
window = 1024
window_small = 16

@measure
def bench():
    rollingrank.rollingrank(x, window=window)

@measure
def bench_float():
    rollingrank.rollingrank(x.astype('float32'), window=window)

@measure
def bench_nan():
    rollingrank.rollingrank(x_nan, window=window)

@measure
def bench_pct():
    rollingrank.rollingrank(x, window=window, pct=True)

def rollingrank_pandas(x, window=None):
    def to_rank(x):
        # result[i] is the rank of x[i] in x
        x = x.values
        return np.sum(x < x[-1])
    return pd.Series(x).rolling(window).apply(to_rank).values

def rollingrank_pandas2(x, window=None):
    def to_rank(x):
        # result[i] is the rank of x[i] in x
        return x.values.argsort().argsort()[-1]
    return pd.Series(x).rolling(window).apply(to_rank).values

@measure
def bench_pandas():
    rollingrank_pandas(x_small, window=window_small)

@measure
def bench_pandas2():
    rollingrank_pandas2(x_small, window=window_small)

bench()
bench_float()
bench_nan()
bench_pct()
bench_pandas()
bench_pandas2()
