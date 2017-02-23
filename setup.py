from setuptools import setup, find_packages
# To use a consistent encoding
from codecs import open
from os import path


here = path.abspath(path.dirname(__file__))
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='conceptnetwork',
    version='0.0.8',

    description='TensorFlow&CloudML helper function for complex data processing.',
    long_description=long_description,

    url='https://github.com/Fematich/conceptnetwork',

    # Author details
    author='Matthias Feys',
    author_email='matthiasfeys@gmail.com',

    packages=find_packages(exclude=['examples','docs', 'tests']),

    install_requires=['tensorflow==1.0.0'],
)