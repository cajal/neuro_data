#!/usr/bin/env python
from setuptools import setup, find_packages
from os import path

here = path.abspath(path.dirname(__file__))

setup(
    name='neuro_data',
    version='0.0.0',
    description='Neuroscience and Machine Learning at Tolias lab @ Baylor College of Medicine',
    author='Fabian Sinz',
    author_email='sinz@bcm.edu',
    url='https://github.com/cajal/neuro_data',
    packages=find_packages(exclude=[]),
    install_requires=['numpy', 'tqdm', 'datajoint', 'attorch', 'pandas', 'h5py'],
)
