from unittest import TestCase
import numpy as np
import pandas as pd
import rollingrank
import random

class TestRci(TestCase):
    def test_normal_case(self):
        x = np.array([0.1, 0.2, 0.3, 0.4, 0.35, 0.25, 0.15])
        y = rollingrank.rci(x, window=3)
        y_ref = rollingrank.rci_reference(x, window=3)
        np.testing.assert_array_equal(y, y_ref)

    def test_float16(self):
        x = np.array([0.1, 0.2, 0.3, 0.4, 0.35, 0.25, 0.15])
        y = rollingrank.rci(x.astype(np.float16), window=3)
        y_ref = rollingrank.rci(x, window=3)
        np.testing.assert_array_equal(y, y_ref)

    def test_method_average(self):
        x = np.array([0.1, 0.1])
        y = rollingrank.rci(x, window=2)
        np.testing.assert_array_equal(y, [np.nan, 0])

    def test_window1(self):
        x = np.array([0.1, 0.2, 0.3, 0.25, 0.1, 0.2, 0.3])
        y = rollingrank.rci(x, window=1)
        np.testing.assert_array_equal(y, [0, 0, 0, 0, 0, 0, 0])

    def test_rollingrank_large_window(self):
        x = np.array([0.1, 0.2, 0.3, 0.25, 0.1, 0.2, 0.3])
        y = rollingrank.rci(x, window=8)
        np.testing.assert_array_equal(y, [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])

    def test_nan(self):
        x = np.array([1, np.nan, 2, np.nan, 3])
        y = rollingrank.rci(x, window=3)
        np.testing.assert_array_equal(y, [np.nan, np.nan, 1, np.nan, 1])

    def test_nan_window1(self):
        x = np.array([1, np.nan, 2])
        y = rollingrank.rci(x, window=1)
        np.testing.assert_array_equal(y, [0, np.nan, 0])

    def test_list_input(self):
        x = [0.1, 0.2, 0.3, 0.4, 0.35, 0.25, 0.15]
        y = rollingrank.rci(x, window=3)
        y_ref = rollingrank.rci(np.array(x), window=3)
        np.testing.assert_array_equal(y, y_ref)

    def test_pandas_series_input(self):
        x = np.array([0.1, 0.2, 0.3, 0.2, 0.1, 0.2, 0.3])
        y = rollingrank.rci(pd.Series(x), window=3)
        y_ref = rollingrank.rci(x, window=3)
        np.testing.assert_array_equal(y, y_ref)

    def test_parallel(self):
        x = np.random.rand(2 ** 20)
        y = rollingrank.rci(x, window=3, n_jobs=1)
        y_parallel = rollingrank.rci(x, window=3)
        np.testing.assert_array_equal(y_parallel, y)

    def test_random_test(self):
        for i in range(10):
            n = random.randint(1, 2 ** 20)
            w = random.randint(1, 2 ** 8)
            x = np.random.rand(n)
            y = rollingrank.rci(x, window=w, n_jobs=1)
            y_parallel = rollingrank.rci(x, window=w)
            np.testing.assert_array_equal(y_parallel, y)
