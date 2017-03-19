#!/usr/bin/python

from setuptools import find_packages
from setuptools import setup

# ,'git+https://github.com/Fematich/conceptnetwork.git']
REQUIRED_PACKAGES = []

setup(
    name='minimalnetwork',
    version='0.1',
    author='Matthias Feys',
    author_email='matthiasfeys@gmail.com',
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    package_data={
        'utils': ['*'],
        # 'config': ['*']
    },
    description='Example of Minimal Network',
    requires=[])
