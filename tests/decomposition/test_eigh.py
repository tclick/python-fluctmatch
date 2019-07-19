# fluctmatch --- https://github.com/tclick/python-fluctmatch
# Copyright (c) 2015-2017 The fluctmatch Development Team and contributors
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

import numpy as np
from numpy import testing

from fluctmatch.decomposition.eigh import Eigh

# Constants
X: np.ndarray = np.array([
    [1, 2, 0],
    [2, 4, 0],
    [0, 0, 3]
])
L: np.ndarray = np.array([5., 3., 0.])
V: np.ndarray = np.array([
    [np.sqrt(0.2), 0, np.sqrt(0.8)],
    [np.sqrt(0.8), 0, -np.sqrt(0.2)],
    [0, 1, 0]
])


def test_eigh():
    eigh = Eigh()
    Vtest = eigh.fit_transform(X)
    testing.assert_almost_equal(Vtest, V, decimal=6)
    testing.assert_almost_equal(eigh.eigenvalues_, L, decimal=6)
