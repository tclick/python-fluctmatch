# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding: utf-8 -*-
# vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4
#
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
N_COMPONENTS_ = 3


def test_full():
    svd = SVD(svd_solver="full")
    Utest = svd.fit_transform(X)
    testing.assert_array_almost_equal(Utest, US, decimal=6)
    testing.assert_array_almost_equal(svd.singular_values_, S, decimal=6)
    testing.assert_array_almost_equal(svd.components_, VT, decimal=6)


def test_randomized():
    svd = SVD(svd_solver="randomized")
    Utest = svd.fit_transform(X)
    testing.assert_array_almost_equal(Utest, US, decimal=6)
    testing.assert_array_almost_equal(svd.singular_values_, S, decimal=6)
    testing.assert_array_almost_equal(svd.components_, VT, decimal=6)


def test_trunc_randomized():
    svd = SVD(n_components=N_COMPONENTS_, svd_solver="randomized")
    Utest = svd.fit_transform(X)
    testing.assert_array_almost_equal(Utest, US[:, :N_COMPONENTS_], decimal=6)
    testing.assert_array_almost_equal(svd.singular_values_, S[:N_COMPONENTS_], decimal=6)
    testing.assert_array_almost_equal(svd.components_, VT[:N_COMPONENTS_], decimal=6)


def test_trunc_arpack():
    svd = SVD(n_components=N_COMPONENTS_, svd_solver="arpack")
    Utest = svd.fit_transform(X)
    testing.assert_array_almost_equal(Utest, US[:, :N_COMPONENTS_], decimal=6)
    testing.assert_array_almost_equal(svd.singular_values_, S[:N_COMPONENTS_], decimal=6)
    testing.assert_array_almost_equal(svd.components_, VT[:N_COMPONENTS_], decimal=6)
