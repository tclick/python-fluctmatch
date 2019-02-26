#  python-fluctmatch -
#  Copyright (c) 2019 Timothy H. Click, Ph.D.
# 
#  All rights reserved.
# 
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:
# 
#  Redistributions of source code must retain the above copyright notice, this
#  list of conditions and the following disclaimer.
# 
#  Redistributions in binary form must reproduce the above copyright notice,
#  this list of conditions and the following disclaimer in the documentation
#  and/or other materials provided with the distribution.
# 
#  Neither the name of the author nor the names of its contributors may be used
#  to endorse or promote products derived from this software without specific
#  prior written permission.
# 
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS “AS IS”
#  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
#  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
#  ARE DISCLAIMED. IN NO EVENT SHALL THE REGENTS OR CONTRIBUTORS BE LIABLE FOR
#  ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
#  DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
#  SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
#  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
#  OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
#  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
#  These tests were taken from the scikit-learn tests for PCA.

import numpy as np
from numpy.testing import assert_almost_equal
from numpy.testing import assert_array_almost_equal
from numpy.testing import assert_greater
from numpy.testing import assert_no_warnings

from sklearn import datasets
from fluctmatch.decomposition.svd import SVD 
from sklearn.decomposition.pca import _infer_dimension_

iris = datasets.load_iris()
solver_list = ['full', 'arpack', 'randomized', 'auto']


def test_svd():
    # PCA on dense arrays
    X = iris.data
    n_comp = X.shape[1]

    svd = SVD()

    X_r = svd.fit(X).transform(X)
    np.testing.assert_equal(X_r.shape[1], n_comp)

    X_r2 = svd.fit_transform(X)
    assert_array_almost_equal(X_r, X_r2)

    X_r = svd.transform(X)
    X_r2 = svd.fit_transform(X)
    assert_array_almost_equal(X_r, X_r2)

    # Test get_covariance and get_precision
    cov = svd.get_covariance()
    precision = svd.get_precision()
    assert_array_almost_equal(np.dot(cov, precision),
                              np.eye(X.shape[1]), 12)

    # test explained_variance_ratio_ == 1 with all components
    svd = SVD()
    svd.fit(X)
    assert_almost_equal(svd.explained_variance_ratio_.sum(), 1.0, 3)


def test_no_empty_slice_warning():
    # test if we avoid numpy warnings for computing over empty arrays
    n_components = 10
    n_features = n_components + 2  # anything > n_comps triggered it in 0.16
    X = np.random.uniform(-1, 1, size=(n_components, n_features))
    svd = SVD()
    assert_no_warnings(svd.fit, X)


def test_svd_inverse():
    # Test that the projection of data can be inverted
    rng = np.random.RandomState(0)
    n, p = 50, 3
    X = rng.randn(n, p)  # spherical data
    X[:, 1] *= .00001  # make middle component relatively small
    X += [5, 4, 3]  # make a large mean

    # same check that we can find the original data from the transformed
    # signal (since the data is almost of rank n_components)
    svd = SVD().fit(X)
    Y = svd.transform(X)
    Y_inverse = svd.inverse_transform(Y)
    assert_almost_equal(X, Y_inverse, decimal=3)


def test_infer_dim_2():
    n, p = 1000, 5
    rng = np.random.RandomState(0)
    X = rng.randn(n, p) * .1
    X[:10] += np.array([3, 4, 5, 1, 2])
    X[10:20] += np.array([6, 0, 7, 2, -1])
    svd = SVD()
    svd.fit(X)
    spect = svd.explained_variance_
    assert_greater(_infer_dimension_(spect, n, p), 1)


def test_infer_dim_3():
    n, p = 100, 5
    rng = np.random.RandomState(0)
    X = rng.randn(n, p) * .1
    X[:10] += np.array([3, 4, 5, 1, 2])
    X[10:20] += np.array([6, 0, 7, 2, -1])
    X[30:40] += 2 * np.array([-1, 1, -1, 1, -1])
    svd = SVD()
    svd.fit(X)
    spect = svd.explained_variance_
    assert_greater(_infer_dimension_(spect, n, p), 2)


def test_svd_float_dtype_preservation():
    # Ensure that PCA does not upscale the dtype when input is float32
    X_64 = np.random.RandomState(0).rand(1000, 4).astype(np.float64)
    X_32 = X_64.astype(np.float32)

    svd_64 = SVD().fit(X_64)
    svd_32 = SVD().fit(X_32)

    assert svd_64.components_.dtype == np.float64
    assert svd_32.components_.dtype == np.float32
    assert svd_64.transform(X_64).dtype == np.float64
    assert svd_32.transform(X_32).dtype == np.float32

    # decimal=5 fails on mac with scipy = 1.1.0
    assert_array_almost_equal(svd_64.components_, svd_32.components_,
                              decimal=4)


def test_svd_int_dtype_usvdst_to_double():
    # Ensure that all int types will be usvdst to float64
    X_i64 = np.random.RandomState(0).randint(0, 1000, (1000, 4))
    X_i64 = X_i64.astype(np.int64)
    X_i32 = X_i64.astype(np.int32)

    svd_64 = SVD().fit(X_i64)
    svd_32 = SVD().fit(X_i32)

    assert svd_64.components_.dtype == np.float64
    assert svd_32.components_.dtype == np.float64
    assert svd_64.transform(X_i64).dtype == np.float64
    assert svd_32.transform(X_i32).dtype == np.float64

    assert_array_almost_equal(svd_64.components_, svd_32.components_,
                              decimal=5)
