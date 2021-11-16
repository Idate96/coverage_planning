from distutils.core import setup
from catkin_pkg.python_setup import generate_distutils_setup

setup_args = generate_distutils_setup(
    packages = ['coverage_planning'],
    package_dir = {'': 'cpp'}
)

setup(**setup_args)

# from setuptools import setup, find_packages
#
# setup(name='cpp', version='0.0', packages=find_packages())