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
import logging
from typing import Optional, Tuple, Union

import numpy as np
from scipy import linalg
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition.base import _BasePCA
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.extmath import fast_logdet, svd_flip

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
        self.components_: np.ndarray = None
        self.singular_values_: np.ndarray = None
        self.explained_variance_: np.ndarray = None
        self.explained_variance_ratio_: np.ndarray = None
        self.n_features_: int = None
        self.n_samples_: int = None

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
        self._fit(X)
        return self

    def _fit(self, X: np.ndarray):
        X: np.ndarray = check_array(X, dtype=[np.float64, np.float32],  copy=self.copy)
        n_samples, n_features = X.shape

        U, S, Vt = linalg.svd(X, full_matrices=False)
        U, Vt = svd_flip(U, Vt)

        # Get variance explained by singular values
        explained_variance_: np.ndarray = (S ** 2) / (n_samples - 1)
        total_var: float = explained_variance_.sum()
        explained_variance_ratio_: np.ndarray = explained_variance_ / total_var
        singular_values_: np.ndarray = S.copy()  # Store the singular values.

        self.n_samples_, self.n_features_ = n_samples, n_features
        self.components_: np.ndarray = Vt
        self.singular_values_: np.ndarray = singular_values_
        self.explained_variance_: np.ndarray = explained_variance_
        self.explained_variance_ratio_: np.ndarray = explained_variance_ratio_

        return U, S, Vt

    def transform(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self, ['singular_values_', 'components_'],
                        all_or_any=all)
        U: np.ndarray = X.dot(self.components_.T) / self.singular_values_
        return U

    def fit_transform(self, X: np.ndarray, y=None) -> np.ndarray:
        U, _, _ = self._fit(X)
        return U

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self, ['singular_values_', 'components_'],
                        all_or_any=all)
        X_transformed: np.ndarray = self.singular_values_ * self.components_.T
        X_transformed: np.ndarray = X.dot(X_transformed.T)
        return X_transformed
