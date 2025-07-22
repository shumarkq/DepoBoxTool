"""
Setup script for DepoBoxTool v2.0 - Refactored and Enhanced

This package provides a comprehensive framework for testing dry deposition
algorithms across various land use categories, designed to support CMAQ
model improvements.
"""

from __future__ import print_function
try:
    from setuptools import setup, find_packages
except ImportError:
    from distutils.core import setup, find_packages

import os

# Read long description from README
def read_long_description():
    here = os.path.abspath(os.path.dirname(__file__))
    readme_path = os.path.join(here, 'README.md')
    if os.path.exists(readme_path):
        with open(readme_path, 'r', encoding='utf-8') as f:
            return f.read()
    return "DepoBoxTool - A comprehensive tool for testing dry deposition algorithms"

setup(
    name='DepoBoxTool',
    version='2.0.0',
    author='Qian Shu',
    author_email='shumarkq@gmail.com',
    maintainer='Qian Shu',
    maintainer_email='shumarkq@gmail.com',
    description='A comprehensive tool for testing dry deposition algorithms',
    long_description=read_long_description(),
    long_description_content_type='text/markdown',
    url='https://github.com/shumarkq/DepoBoxTool',
    project_urls={
        'Documentation': 'https://github.com/shumarkq/DepoBoxTool',
        'Source': 'https://github.com/shumarkq/DepoBoxTool',
        'Tracker': 'https://github.com/shumarkq/DepoBoxTool/issues',
    },
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    python_requires='>=3.7',
    install_requires=[
        'numpy>=1.19.0',
        'scipy>=1.5.0',
        'pandas>=1.1.0',
        'matplotlib>=3.3.0',
        'seaborn>=0.11.0',
    ],
    extras_require={
        'dev': [
            'pytest>=6.0',
            'pytest-cov>=2.10',
            'black>=21.0',
            'flake8>=3.8',
            'mypy>=0.910',
            'sphinx>=4.0',
            'sphinx-rtd-theme>=1.0',
        ],
        'test': [
            'pytest>=6.0',
            'pytest-cov>=2.10',
        ],
    },
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Topic :: Scientific/Engineering :: Atmospheric Science',
        'Topic :: Scientific/Engineering :: Physics',
    ],
    keywords='atmospheric-science deposition aerosol cmaq air-quality',
    entry_points={
        'console_scripts': [
            'depobox-example=examples.modern_deposition_example:main',
        ],
    },
    include_package_data=True,
    zip_safe=False,
)