#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from setuptools import setup, Extension

class get_pybind_include(object):
    def __init__(self, user=False):
        self.user = user

    def __str__(self):
        import pybind11
        return pybind11.get_include(self.user)

setup(
    ext_modules=[
        Extension(
            'rollingrank_native',
            ['src/rollingrank.cpp'],
            extra_compile_args=[
                '-std=c++14',
                '-pthread',
            ],
            include_dirs = [
                get_pybind_include(),
                get_pybind_include(user=True),
                os.path.dirname(os.path.abspath(__file__)) + '/deps/cpp-taskflow',
                os.path.dirname(os.path.abspath(__file__)) + '/deps/rank_in_range/include'
            ]
        )
    ],
)
