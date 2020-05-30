import numpy as np
import pandas as pd
import rollingrank

from pythonbenchmark import measure

x = np.random.rand(1024 * 1024)
x_nan = x.copy()
x_nan[np.arange(0, x.size, 2)] = np.nan
x_small = np.random.rand(16 * 1024)
window = 128
window_small = 16

@measure
def bench():
    rollingrank.rci(x, window=window)

@measure
def bench_single():
    rollingrank.rci(x, window=window, n_jobs=1)

@measure
def bench_float():
    rollingrank.rci(x.astype('float32'), window=window)

@measure
def bench_nan():
    rollingrank.rci(x_nan, window=window)

@measure
def bench_reference():
    rollingrank.rci_reference(x_small, window=window_small)

bench()
bench_single()
bench_float()
bench_nan()
bench_reference()
