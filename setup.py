#!/usr/bin/env python3
# Copyright GC-DPR authors.
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from setuptools import setup

with open('README.md') as f:
    readme = f.read()

setup(
    name='gc-dpr',
    version='0.1.0',
    description='Gradient Cached Dense Passage Retrieval',
    url='',  # TODO
    classifiers=[
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.8',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    long_description=readme,
    long_description_content_type='text/markdown',
    setup_requires=[
        'setuptools>=18.0',
    ],
    install_requires=[
        'cython',
        'faiss-cpu>=1.6.1',
        'filelock',
        'numpy',
        'regex',
        'torch>=1.6.0',
        'transformers==4.12.5',
        'tqdm>=4.27',
        'wget',
        'spacy>=2.1.8',
    ],
)
