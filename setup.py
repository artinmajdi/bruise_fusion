"""Setup configuration for the Bruise Fusion package.

This module handles the installation and packaging of the Bruise Fusion library,
which provides spatial-frequency fusion of white-light and ALS images for bruise analysis.
"""

from setuptools import setup, find_packages
import os
import codecs


# Get the absolute path to the directory containing setup.py
here = os.path.abspath(os.path.dirname(__file__))

# Read the README file
with codecs.open(os.path.join(here, "README.md"), encoding="utf-8") as fh:
    long_description = fh.read()

# Define requirements based on pyproject.toml
requirements = [
    "opencv-python>=4.12.0",
    "numpy>=2.2.0",
    "matplotlib>=3.10.0",
    "streamlit>=1.28.0",
    "Pillow>=9.0.0",
    "pandas>=2.3.0",
    "seaborn>=0.13.0",
    "watchdog>=6.0.0",
    "pywavelets>=1.8.0",
    "scikit-image>=0.25.2",
    "rich>=14.1.0",
    "rawpy>=0.21.0",
    "imageio>=2.37.0",
    "python-dotenv>=0.9.9",
]

# Package metadata
PACKAGE_NAME = "bruise-fusion"
VERSION      = "0.2.1"
AUTHOR       = "Artin Majdi"
AUTHOR_EMAIL = "mmajdi@gmu.edu"
DESCRIPTION  = "Spatial-frequency fusion of white-light and ALS images for bruise analysis"
URL          = "https://github.com/artinmajdi/bruise_fusion"
LICENSE      = "MIT"

# Classifiers for PyPI
CLASSIFIERS = [
    "Development Status :: 4 - Beta",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: MIT License",
    "Operating System :: OS Independent",
    "Natural Language :: English",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3 :: Only",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Programming Language :: Python :: Implementation :: CPython",
    "Topic :: Scientific/Engineering :: Image Processing",
    "Topic :: Scientific/Engineering :: Medical Science Apps.",
]


setup(
    name=PACKAGE_NAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type="text/markdown",
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    url=URL,
    packages=["src"],
    include_package_data=True,
    python_requires=">=3.10",
    install_requires=requirements,
    classifiers=CLASSIFIERS,
    entry_points={
        'console_scripts': [
            'bfuse=src.dashboard:main',
        ],
    },
    keywords=[
        "image-processing",
        "medical-imaging",
        "bruise-analysis",
        "image-fusion",
        "spatial-frequency",
        "als-imaging",
    ],
    project_urls={
        "Homepage": URL,
        "Repository": f"{URL}.git",
        "Issues": f"{URL}/issues",
        "Source Code": URL,
    },
)
