#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, Extension

setup(
    ext_modules=[
        Extension(
            'rollingrank',
            ['src/rollingrank.cpp'],
            extra_compile_args=['-std=c++11'],
        )
    ],
)
