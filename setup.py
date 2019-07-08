#!/usr/bin/env python
#
# fluctmatch --- https://github.com/tclick/python-fluctmatch
# Copyright (c) 2013-2017 The fluctmatch Development Team and contributors
# (see the file AUTHORS for the full list of names)
#
# Released under the New BSD license.
#
# Please cite your use of fluctmatch in published work:
#
# Timothy H. Click, Nixon Raj, and Jhih-Wei Chu.
# Calculation of Enzyme Fluctuograms from All-Atom Molecular Dynamics
# Simulation. Meth Enzymology. 578 (2016), 327-342,
# doi:10.1016/bs.mie.2016.05.024.
#
import re
import sys
from glob import glob
from os.path import basename
from os.path import splitext
from pathlib import Path

from setuptools import find_packages
from setuptools import setup

# NOTE: keep in sync with DNApersist.__version__ in version.py
RELEASE = "4.0.0-dev"
is_release = "dev" not in RELEASE

# Make sure I have the right Python version.
major, minor = sys.version_info[:2]
if (major, minor) < (3, 6):
    print(
        f"dnapersist requires Python 3.6 or better. "
        f"Python {major:d}.{minor:d} detected"
    )
    print("Please upgrade your version of Python.")
    sys.exit(-1)


def read(*names, **kwargs):
    return open(
        Path().joinpath(Path(__file__).parent, *names),
        encoding=kwargs.get("encoding", "utf8"),
    ).read()


if __name__ == "__main__":
    setup(
        name="fluctmatch",
        version="3.4.1",
        license="BSD license",
        description="Elastic network model using fluctuation matching.",
        long_description="%s\n%s"
                         % (
                             re.compile("^.. start-badges.*^.. end-badges",
                                        re.M | re.S).sub(
                                 "", read("README.rst")
                             ),
                             re.sub(":[a-z]+:`~?(.*?)`", r"``\1``",
                                    read("CHANGELOG.rst")),
                         ),
        author="Timothy Click",
        author_email="tclick@nctu.edu.tw",
        url="https://www.github.com/tclick/python-fluctmatch",
        packages=find_packages("src"),
        package_dir={"": "src"},
        py_modules=[splitext(basename(path))[0] for path in glob("src/*.py")],
        include_package_data=True,
        zip_safe=False,
        classifiers=[
            # complete classifier list: http://pypi.python.org/pypi?%3Aaction=list_classifiers
            "Development Status :: 5 - Production/Stable",
            "Intended Audience :: Developers",
            "License :: OSI Approved :: BSD License",
            "Operating System :: Unix",
            "Operating System :: POSIX",
            "Operating System :: Microsoft :: Windows",
            "Programming Language :: Python",
            "Programming Language :: Python :: 3.6",
            "Programming Language :: Python :: 3.7",
            "Programming Language :: Python :: Implementation :: CPython",
            "Programming Language :: Python :: Implementation :: PyPy",
            "Topic :: Scientific/Engineering :: Chemistry",
        ],
        keywords=["elastic network model", "fluctuation matching"],
        install_requires=[
            "click",
            "MDAnalysis",
            "numpy",
            "pandas",
            "scipy",
            "scikit-learn",
        ],
        extras_require={
            "dev": ["MDAnalysisTests", "pytest", "tox", "coverage", "coveralls"]
        },
        entry_points={"console_scripts": ["fluctmatch = fluctmatch.cli:main"]},
    )
