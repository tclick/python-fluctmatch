# -*- coding: utf-8 -*-
#
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
#  Timothy H. Click, Nixon Raj, and Jhih-Wei Chu.
#  Simulation. Meth Enzymology. 578 (2016), 327-342,
#  Calculation of Enzyme Fluctuograms from All-Atom Molecular Dynamics
#  doi:10.1016/bs.mie.2016.05.024.

import logging

import numpy as np
from scipy import linalg
from sklearn.decomposition.base import _BasePCA
from sklearn.utils import check_array, safe_sqr
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.extmath import svd_flip

logger = logging.getLogger(__name__)


class SVD(_BasePCA):
    """Dimensionality reduction using singular-value decomposition.

    This transformer performs linear dimensionality reduction by means of
    singular value decomposition (SVD). Contrary to PCA, this estimator
    does not center the data before computing the singular value decomposition.

    Parameters
    ----------
    copy : bool (default True)
        If False, data passed to fit are overwritten and running
        fit(X).transform(X) will not yield the expected results,
        use fit_transform(X) instead.

    Attributes
    ----------
    components_ : array, shape (n_samples, n_features)

    explained_variance_ : array, shape (n_samples,)
        The variance of the training samples transformed by a projection to
        each component.

    explained_variance_ratio_ : array, shape (n_samples,)
        Percentage of variance explained by each of the selected components.

    singular_values_ : array, shape (n_samples,)
        The singular values corresponding to each of the selected components.
        The singular values are equal to the 2-norms of the ``n_components``
        variables in the lower-dimensional space.

    Examples
    --------
    >>> import numpy as np
    >>> from fluctmatch.decomposition.svd import SVD
    >>> X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
    >>> svd = SVD()
    >>> svd.fit(X)
    SVD(copy=True, iterated_power='auto', random_state=None, svd_solver='auto')
    >>> print(svd.explained_variance_ratio_)  # doctest: +ELLIPSIS
    [0.9924... 0.0075...]
    >>> print(svd.singular_values_)  # doctest: +ELLIPSIS
    [6.30061... 0.54980...]

    >>> svd = SVD(n_svd_solver='full')
    >>> svd.fit(X)                 # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
    PCA(copy=True, iterated_power='auto', n_components=2, random_state=None,
      svd_solver='full', tol=0.0, whiten=False)
    >>> print(svd.explained_variance_ratio_)  # doctest: +ELLIPSIS
    [0.9924... 0.00755...]
    >>> print(svd.singular_values_)  # doctest: +ELLIPSIS
    [6.30061... 0.54980...]

    Notes
    -----
    SVD suffers from a problem called "sign indeterminacy", which means the
    sign of the ``components_`` and the output from transform depend on the
    algorithm and random state. To work around this, fit instances of this
    class to data once, then keep the instance around to do transformations.
    """
    def __init__(self, copy: bool=True):
        self.copy: bool = copy

    def get_covariance(self) -> np.ndarray:
        """Compute data covariance with the generative model.

        ``cov = components_.T * S**2 * components_ + sigma2 * eye(n_features)``
        where  S**2 contains the explained variances, and sigma2 contains the
        noise variances.

        Returns
        -------
        cov : array, shape=(n_features, n_features)
            Estimated covariance of data.
        """
        components_: np.ndarray = self.components_
        cov: np.ndarray = np.dot(components_.T * self.explained_variance_,
                                 components_)
        return cov

    def get_precision(self) -> np.ndarray:
        """Compute data precision matrix with the generative model.

        Equals the inverse of the covariance but computed with
        the matrix inversion lemma for efficiency.

        Returns
        -------
        precision : array, shape=(n_features, n_features)
            Estimated precision of data.
        """
        n_features = self.components_.shape[1]

        # handle corner cases first
        return linalg.inv(self.get_covariance())

    def fit(self, X: np.ndarray, y=None) -> "SVD":
        """Fit LSI model on training data X.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Training data.

        y : Ignored

        Returns
        -------
        self : object
            Returns the transformer object.
        """
        self.fit_transform(X)
        return self

    def fit_transform(self, X: np.ndarray, y=None) -> np.ndarray:
        """Fit LSI model to X and perform SVD on X.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Training data.

        y : Ignored

        Returns
        -------
        X_new : array, shape (n_samples, n_features)
            Reduced version of X. This will always be a dense array.
        """
        X: np.ndarray = check_array(X, dtype=[np.float64, np.float32],  copy=self.copy)
        n_samples, n_features = X.shape

        U, Sigma, VT = linalg.svd(X, full_matrices=False)
        U, VT = svd_flip(U, VT)

        self.n_components = Sigma.size
        self.components_ = VT

        # Calculate explained variance & explained variance ratio
        X_transformed = U * Sigma
        self.explained_variance_ = exp_var = np.var(X_transformed, axis=0)
        full_var = np.var(X, axis=0).sum()
        self.explained_variance_ratio_ = exp_var / full_var
        self.singular_values_ = Sigma  # Store the singular values.

        return X_transformed

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Perform SVD on X.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            New data.

        Returns
        -------
        X_new : array, shape (n_samples, n_features)
            Reduced version of X. This will always be a dense array.
        """
        check_is_fitted(self, ['singular_values_', 'components_'],
                        all_or_any=all)
        X: np.ndarray = check_array(X, dtype=[np.float64, np.float32],  copy=self.copy)
        return np.dot(X, self.components_.T)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """Transform X back to its original space.

        Returns an array X_original whose transform would be X.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_components)
            New data.

        Returns
        -------
        X_original : array, shape (n_samples, n_features)
            Note that this is always a dense array.
        """
        X: np.ndarray = check_array(X)
        return np.dot(X, self.components_)
