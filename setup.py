# ------------------------------------------------------------------------------
#   python-fluctmatch
#   Copyright (c) 2013-2020 Timothy H. Click, Ph.D.
#
#   All rights reserved.
#
#   Redistribution and use in source and binary forms, with or without
#   modification, are permitted provided that the following conditions are met:
#
#   Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
#   Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
#   Neither the name of the author nor the names of its contributors may be used
#   to endorse or promote products derived from this software without specific
#   prior written permission.
#
#    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS “AS IS”
#    AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
#    IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
#    ARE DISCLAIMED. IN NO EVENT SHALL THE REGENTS OR CONTRIBUTORS BE LIABLE FOR
#    ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
#    DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
#    SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
#    CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
#    LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
#    OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH
#    DAMAGE.
#
#   Timothy H. Click, Nixon Raj, and Jhih-Wei Chu.
#   Simulation. Meth Enzymology. 578 (2016), 327-342,
#   Calculation of Enzyme Fluctuograms from All-Atom Molecular Dynamics
#   doi:10.1016/bs.mie.2016.05.024.
#
# ------------------------------------------------------------------------------

import re
import sys
from glob import glob
from os.path import basename
from os.path import splitext
from pathlib import Path

from setuptools import find_packages
from setuptools import setup

# NOTE: keep in sync with DNApersist.__version__ in version.py
RELEASE: str = "4.0.0-dev"
is_release: str = "dev" not in RELEASE

# Make sure I have the right Python version.
major, minor = sys.version_info[:2]
if (major, minor) < (3, 6):
    print(
        f"fluctmatch requires Python 3.6 or better. "
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
        version="4.0.0",
        license="BSD license",
        description="Elastic network model using fluctuation matching.",
        long_description="%s\n%s"
        % (
            re.compile("^.. start-badges.*^.. end-badges", re.M | re.S).sub(
                "", read("README.rst")
            ),
            re.sub(":[a-z]+:`~?(.*?)`", r"``\1``", read("CHANGELOG.rst")),
        ),
        author="Timothy Click",
        author_email="tclick@tabor.edu",
        url="https://www.github.com/tclick/python-fluctmatch",
        packages=find_packages("src"),
        package_dir={"": "src"},
        py_modules=[splitext(basename(path))[0] for path in glob("src/*.py")],
        include_package_data=True,
        zip_safe=False,
        classifiers=[
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
            "matplotlib",
            "numpy",
            "pandas",
            "scipy",
            "scikit-learn",
        ],
        extras_require=dict(
            dev=["MDAnalysisTests", "pytest", "tox", "coverage", "coveralls"]
        ),
        entry_points={"console_scripts": ["fluctmatch = fluctmatch.cli:main"]},
    )
