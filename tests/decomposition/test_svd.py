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

import numpy as np
from numpy import testing
from sklearn.utils.extmath import svd_flip

from fluctmatch.decomposition.svd import SVD

# Constants
X: np.ndarray = np.array([
    [1, 0, 0, 0, 2],
    [0, 0, 3, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 2, 0, 0, 0]
])
U: np.ndarray = np.array([
    [0, 1, 0, 0],
    [1, 0, 0, 0],
    [0, 0, 0, -1],
    [0, 0, 1, 0]
])
S: np.ndarray = np.array([3, np.sqrt(5), 2, 0])
VT: np.ndarray = np.array([
    [0, 0, 1, 0, 0],
    [np.sqrt(0.2), 0, 0, 0, np.sqrt(0.8)],
    [0, 1, 0, 0, 0],
    [0, 0, 0, 1, 0]
])
U, VT = svd_flip(U, VT)
US: np.ndarray = U * S
N_COMPONENTS_: int = 3
DECIMAL: int = 4


def test_full():
    svd: SVD = SVD(svd_solver="full")
    Utest: np.ndarray = svd.fit_transform(X)
    testing.assert_array_almost_equal(Utest, US, decimal=DECIMAL)
    testing.assert_array_almost_equal(svd.singular_values_, S, decimal=DECIMAL)
    testing.assert_array_almost_equal(svd.components_, VT, decimal=DECIMAL)


def test_randomized():
    svd: SVD = SVD(svd_solver="randomized")
    Utest: np.ndarray = svd.fit_transform(X)
    testing.assert_array_almost_equal(Utest, US, decimal=DECIMAL)
    testing.assert_array_almost_equal(svd.singular_values_, S, decimal=DECIMAL)
    testing.assert_array_almost_equal(svd.components_, VT, decimal=DECIMAL)


def test_trunc_randomized():
    svd: SVD = SVD(n_components=N_COMPONENTS_, svd_solver="randomized")
    Utest: np.ndarray = svd.fit_transform(X)
    testing.assert_array_almost_equal(Utest, US[:, :N_COMPONENTS_],
                                      decimal=DECIMAL)
    testing.assert_array_almost_equal(svd.singular_values_, S[:N_COMPONENTS_],
                                      decimal=DECIMAL)
    testing.assert_array_almost_equal(svd.components_, VT[:N_COMPONENTS_],
                                      decimal=DECIMAL)


def test_trunc_arpack():
    svd: SVD = SVD(n_components=N_COMPONENTS_, svd_solver="arpack")
    Utest: np.ndarray = svd.fit_transform(X)
    testing.assert_array_almost_equal(Utest, US[:, :N_COMPONENTS_],
                                      decimal=DECIMAL)
    testing.assert_array_almost_equal(svd.singular_values_, S[:N_COMPONENTS_],
                                      decimal=DECIMAL)
    testing.assert_array_almost_equal(svd.components_, VT[:N_COMPONENTS_],
                                      decimal=DECIMAL)
