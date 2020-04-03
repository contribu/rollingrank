from unittest import TestCase
import numpy as np
import rollingrank

class TestRollingrank(TestCase):
    def test_normal_case(self):
        x = np.array([0.1, 0.2, 0.3, 0.25, 0.1, 0.2, 0.3])
        y = rollingrank.rollingrank(x, window=3)
        np.testing.assert_array_equal(y, [np.nan, np.nan, 2, 1, 0, 1, 2])

    def test_method_default(self):
        x = np.array([0.1, 0.1])
        y = rollingrank.rollingrank(x, window=2)
        np.testing.assert_array_equal(y, [np.nan, 0.5])

    def test_method_average(self):
        x = np.array([0.1, 0.1])
        y = rollingrank.rollingrank(x, window=2, method='average')
        np.testing.assert_array_equal(y, [np.nan, 0.5])

    def test_method_min(self):
        x = np.array([0.1, 0.1])
        y = rollingrank.rollingrank(x, window=2, method='min')
        np.testing.assert_array_equal(y, [np.nan, 0])

    def test_method_max(self):
        x = np.array([0.1, 0.1])
        y = rollingrank.rollingrank(x, window=2, method='max')
        np.testing.assert_array_equal(y, [np.nan, 1])

    def test_method_first(self):
        x = np.array([0.1, 0.1])
        y = rollingrank.rollingrank(x, window=2, method='first')
        np.testing.assert_array_equal(y, [np.nan, 1])

    def test_window1(self):
        x = np.array([0.1, 0.2, 0.3, 0.25, 0.1, 0.2, 0.3])
        y = rollingrank.rollingrank(x, window=1)
        np.testing.assert_array_equal(y, [0, 0, 0, 0, 0, 0, 0])

    def test_rollingrank_same_window(self):
        x = np.array([0.1, 0.2, 0.3, 0.25, 0.1, 0.2, 0.3])
        y = rollingrank.rollingrank(x, window=7)
        np.testing.assert_array_equal(y, [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 5.5])

    def test_rollingrank_large_window(self):
        x = np.array([0.1, 0.2, 0.3, 0.25, 0.1, 0.2, 0.3])
        y = rollingrank.rollingrank(x, window=8)
        np.testing.assert_array_equal(y, [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])

    def test_rollingrank_pct(self):
        x = np.array([0.1, 0.2, 0.3, 0.25, 0.1, 0.2, 0.3])
        y = rollingrank.rollingrank(x, window=3, pct=True)
        np.testing.assert_array_equal(y, [np.nan, np.nan, 1, 0.5, 0, 0.5, 1])

    def test_complex_case(self):
        x = np.array([0.1, 0.2, 0.3, 0.2, 0.1, 0.2, 0.3])
        y = rollingrank.rollingrank(x, window=3)
        np.testing.assert_array_equal(y, [np.nan, np.nan, 2, 0.5, 0, 1.5, 2])

