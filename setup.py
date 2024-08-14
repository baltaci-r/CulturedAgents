# -*- coding: utf-8 -*-
from setuptools import setup, find_packages

requires = [
    'pyautogen',
    'datasets',
    'pandas',
    'seaborn',
    'plotly',
    'scikit-learn',
    ]

setup(
        name='conformity',
        description='implementation of paper code',
        author='baltaji',
        url='https://github.com/baltaci-r/CulturedAgents',
        packages=find_packages(),
        install_requires=requires,
        tests_require=requires,
        )
