#!/usr/bin/env python
from setuptools import setup, find_packages
from os import path

here = path.abspath(path.dirname(__file__))

setup(
    name='quickdraw',
    version='0.0.0',
    description='Implementation of ResNet on the Quick, Draw! data set',
    author='Nikos Karantzas',
    author_email='nikolaos.karantzas@bcm.edu',
    url='https://github.com/nkarantzas/QuickDraw',
    packages=find_packages(exclude=[]),
    install_requires=['numpy', 'cairocffi', 'opencv-python', 'torch', 'torchvision', 'tqdm', 'pandas', 'h5py']
)
