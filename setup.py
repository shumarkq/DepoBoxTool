from __future__ import print_function
try:
    from setuptools import setup, find_packages
except ImportError:
    from distutils.core import setup

# Read version from package
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    from DepoBoxTool import __version__, __author__, __email__
except ImportError:
    __version__ = "2.1.0"
    __author__ = "Qian Shu"
    __email__ = "shumarkq@gmail.com"

# Read README for long description
readme_path = os.path.join(os.path.dirname(__file__), 'README.md')
if os.path.exists(readme_path):
    with open(readme_path, 'r', encoding='utf-8') as f:
        long_description = f.read()
else:
    long_description = "A Python tool for atmospheric particle dry deposition modeling."

setup(
    name='DepoBoxTool',
    version=__version__,
    
    # Author information
    author=__author__,
    author_email=__email__,
    maintainer=__author__,
    maintainer_email=__email__,
    
    # Package information
    description='A Python tool for atmospheric particle dry deposition modeling',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/shumarkq/DepoBoxTool',
    
    # Package structure
    package_dir={'': 'src'},
    packages=find_packages(where='src'),
    
    # Dependencies
    install_requires=[
        'numpy>=1.15.0',
        'scipy>=1.0.0',
        'pandas>=0.24.0',
        'matplotlib>=3.0.0'
    ],
    
    # Optional dependencies
    extras_require={
        'dev': [
            'pytest>=4.0.0',
            'pytest-cov>=2.6.0',
            'flake8>=3.6.0',
            'black>=19.0.0'
        ],
        'docs': [
            'sphinx>=2.0.0',
            'sphinx-rtd-theme>=0.4.0'
        ]
    },
    
    # Metadata
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Topic :: Scientific/Engineering :: Atmospheric Science',
        'Topic :: Scientific/Engineering :: Physics'
    ],
    
    keywords='atmospheric modeling, dry deposition, CMAQ, particles, aerosols',
    
    python_requires='>=3.6',
    
    # Entry points for command line tools (optional)
    entry_points={
        'console_scripts': [
            'depobox-example=DepoBoxTool.scripts:run_example',
        ],
    },
    
    # Include additional files
    include_package_data=True,
    package_data={
        'DepoBoxTool': ['*.txt', '*.md'],
    },
    
    # Project URLs
    project_urls={
        'Bug Reports': 'https://github.com/shumarkq/DepoBoxTool/issues',
        'Source': 'https://github.com/shumarkq/DepoBoxTool',
        'Documentation': 'https://github.com/shumarkq/DepoBoxTool/blob/main/README.md',
    }
)