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
import numbers

import numpy as np
from scipy import linalg
from scipy.sparse import issparse
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
from sklearn.decomposition.pca import _infer_dimension_
from sklearn.decomposition.truncated_svd import TruncatedSVD
from sklearn.utils import check_array
from sklearn.utils import check_random_state
from sklearn.utils.extmath import randomized_svd
from sklearn.utils.extmath import stable_cumsum
from sklearn.utils.extmath import svd_flip

logger = logging.getLogger(__name__)


class SVD(BaseEstimator, TransformerMixin):
    """Dimensionality reduction using singular-value decomposition.

    This transformer performs linear dimensionality reduction by means of
    singular value decomposition (SVD). Contrary to PCA, this estimator
    does not center the data before computing the singular value decomposition.

    Parameters
    ----------
    n_components : int, float, None or string
        Number of components to keep.
        if n_components is not set all components are kept::

            n_components == min(n_samples, n_features)

        If ``n_components == 'mle'`` and ``svd_solver == 'full'``, Minka's
        MLE is used to guess the dimension. Use of ``n_components == 'mle'``
        will interpret ``svd_solver == 'auto'`` as ``svd_solver == 'full'``.

        If ``0 < n_components < 1`` and ``svd_solver == 'full'``, select the
        number of components such that the amount of variance that needs to be
        explained is greater than the percentage specified by n_components.

        If ``svd_solver == 'arpack'``, the number of components must be
        strictly less than the minimum of n_features and n_samples.

        Hence, the None case results in::

            n_components == min(n_samples, n_features) - 1

    copy : bool (default True)
        If False, data passed to fit are overwritten and running
        fit(X).transform(X) will not yield the expected results,
        use fit_transform(X) instead.

    svd_solver : string {'auto', 'full', 'arpack', 'randomized'}
        auto :
            the solver is selected by a default policy based on `X.shape` and
            `n_components`: if the input data is larger than 500x500 and the
            number of components to extract is lower than 80% of the smallest
            dimension of the data, then the more efficient 'randomized'
            method is enabled. Otherwise the exact full SVD is computed and
            optionally truncated afterwards.
        full :
            run exact full SVD calling the standard LAPACK solver via
            `scipy.linalg.svd` and select the components by postprocessing
        arpack :
            run SVD truncated to n_components calling ARPACK solver via
            `scipy.sparse.linalg.svds`. It requires strictly
            0 < n_components < min(X.shape)
        randomized :
            run randomized SVD by the method of Halko et al.

        .. versionadded:: 0.18.0

    tol : float >= 0, optional (default .0)
        Tolerance for singular values computed by svd_solver == 'arpack'.

        .. versionadded:: 0.18.0

    iterated_power : int >= 0, or 'auto', (default 'auto')
        Number of iterations for the power method computed by
        svd_solver == 'randomized'.

    random_state : int, RandomState instance or None, optional (default None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`. Used when ``svd_solver`` == 'arpack' or 'randomized'.

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

    References
    ----------
    For svd_solver == 'randomized', see:
    `Halko, N., Martinsson, P. G., and Tropp, J. A. (2011).
    "Finding structure with randomness: Probabilistic algorithms for
    constructing approximate matrix decompositions".
    SIAM review, 53(2), 217-288.` and also
    `Martinsson, P. G., Rokhlin, V., and Tygert, M. (2011).
    "A randomized algorithm for the decomposition of matrices".
    Applied and Computational Harmonic Analysis, 30(1), 47-68.`

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
    def __init__(self, n_components=None, copy=True, svd_solver='auto',
                 tol=0.0, iterated_power='auto', random_state=None):
        self.n_components = n_components
        self.copy = copy
        self.svd_solver = svd_solver
        self.tol = tol
        self.iterated_power = iterated_power
        self.random_state = random_state

    def fit(self, X, y=None):
        """Fit the model with X.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.

        y : Ignored

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        self._fit(X)
        return self

    def transform(self, X, y=None):
        """Fit the model with X and apply the dimensionality reduction on X.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.

        y : Ignored

        Returns
        -------
        X_new : array-like, shape (n_samples, n_components)

        """
        U, S, V = self._fit(X)

        return U

    def _fit(self, X):
        """Dispatch to the right submethod depending on the chosen solver."""

        # Raise an error for sparse input.
        # This is more informative than the generic one raised by check_array.
        X: np.ndarray = check_array(X, dtype=[np.float64, np.float32],
                                    ensure_2d=True, copy=self.copy,
                                    accept_sparse=['csr', 'csc'])

        if issparse(X) and self.n_components == min(X.shape):
            raise ValueError("n_components=%r must be strictly less than "
                             "min(n_samples, n_features)=%r for a sparse matrix. "
                             % (self.n_components, min(X.shape)))

        # Handle n_components==None
        if self.n_components is None:
            if self.svd_solver != "arpack":
                n_components = min(X.shape)
            else:
                n_components = min(X.shape) - 1
        else:
            n_components = self.n_components

        # Handle svd_solver
        self._fit_svd_solver = self.svd_solver
        if self._fit_svd_solver == "auto":
            # Small problem or n_components == 'mle', just call full PCA
            if max(X.shape) <= 500:
                self._fit_svd_solver = "full"
            elif n_components >= 1 and n_components < .8 * min(X.shape):
                self._fit_svd_solver = "randomized"
            elif issparse(X):
                self._fit_svd_solver = "randomized"
            # This is also the case of n_components in (0,1)
            else:
                self._fit_svd_solver = "full"

        # Call different fits for either full or truncated SVD
        if self._fit_svd_solver == "full":
            if issparse(X):
                raise ValueError(
                    "Cannot use %r for a sparse matrix. Select either"
                    "'arpack' or 'randomized'.".format(self._fit_svd_solver))
            return self._fit_full(X, n_components)
        elif self._fit_svd_solver in ["arpack", "randomized"]:
            return self._fit_truncated(X, n_components, self._fit_svd_solver)
        else:
            raise ValueError(f"Unrecognized svd_solver={self._fit_svd_solver}")

    def _fit_full(self, X, n_components):
        """Fit the model by computing full SVD on X"""
        n_samples, n_features = X.shape

        if n_components == 'mle':
            if n_samples < n_features:
                raise ValueError("n_components='mle' is only supported "
                                 "if n_samples >= n_features")
        elif not 0 <= n_components <= min(n_samples, n_features):
            raise ValueError("n_components=%r must be between 0 and "
                             "min(n_samples, n_features)=%r with "
                             "svd_solver='full'"
                             % (n_components, min(n_samples, n_features)))
        elif n_components >= 1:
            if not isinstance(n_components, (numbers.Integral, np.integer)):
                raise ValueError("n_components=%r must be of type int "
                                 "when greater than or equal to 1, "
                                 "was of type=%r"
                                 % (n_components, type(n_components)))

        U, S, V = linalg.svd(X, full_matrices=False)

        # flip eigenvectors' sign to enforce deterministic output
        U, V = svd_flip(U, V)

        components_: np.ndarray = V

        # Get variance explained by singular values
        explained_variance_: np.ndarray = np.square(S) / n_samples
        total_var: np.ndarray = explained_variance_.sum()
        explained_variance_ratio_: np.ndarray = explained_variance_ / total_var
        singular_values_: np.ndarray = S.copy()  # Store the singular values.

        if n_components == 'mle':
            n_components = \
                _infer_dimension_(explained_variance_, n_samples, n_features)
        elif 0 < n_components < 1.0:
            # number of components for which the cumulated explained
            # variance percentage is superior to the desired threshold
            ratio_cumsum = stable_cumsum(explained_variance_ratio_)
            n_components = np.searchsorted(ratio_cumsum, n_components) + 1

        # Compute noise covariance using Probabilistic PCA model
        # The sigma2 maximum likelihood (cf. eq. 12.46)
        if n_components < min(n_features, n_samples):
            self.noise_variance_ = explained_variance_[n_components:].mean()
        else:
            self.noise_variance_ = 0.

        self.n_samples_, self.n_features_ = n_samples, n_features
        self.components_ = components_[:n_components]
        self.n_components_ = n_components
        self.explained_variance_ = explained_variance_[:n_components]
        self.explained_variance_ratio_ = \
            explained_variance_ratio_[:n_components]
        self.singular_values_ = singular_values_[:n_components]

        U = U[:, :self.n_components_]

        # X_new = X * V = U * S * V^T * V = U * S
        U *= S[:self.n_components_]

        return U, S, V

    def _fit_truncated(self, X, n_components, svd_solver):
        """Fit the model by computing truncated SVD (randomized) on X
        """
        n_samples, n_features = X.shape

        random_state = check_random_state(self.random_state)

        if svd_solver == 'arpack' and n_components == min(n_samples, n_features):
            raise ValueError("n_components=%r must be strictly less than "
                             "min(n_samples, n_features)=%r with "
                             "svd_solver='%s'"
                             % (n_components, min(n_samples, n_features),
                                svd_solver))

        if n_components < min(n_samples, n_features):
            tsvd = TruncatedSVD(n_components=n_components,
                                algorithm=svd_solver,
                                random_state=random_state)
            U = tsvd.fit_transform(X)
            S, V = tsvd.singular_values_, tsvd.components_
        else:
            # sign flipping is done inside
            U, S, V = randomized_svd(X, n_components=n_components,
                                     n_iter=self.iterated_power,
                                     flip_sign=True, random_state=random_state)
            U = U[:, :n_components]

            # X_new = X * V = U * S * V^T * V = U * S
            U *= S[:n_components]

        components_: np.ndarray = V

        # Get variance explained by singular values
        explained_variance_: np.ndarray = np.square(S) / n_samples
        total_var: np.ndarray = explained_variance_.sum()
        explained_variance_ratio_: np.ndarray = explained_variance_ / total_var
        singular_values_: np.ndarray = S.copy()  # Store the singular values.

        # Compute noise covariance using Probabilistic PCA model
        # The sigma2 maximum likelihood (cf. eq. 12.46)
        if n_components < min(n_features, n_samples):
            self.noise_variance_ = explained_variance_[n_components:].mean()
        else:
            self.noise_variance_ = 0.

        self.n_samples_, self.n_features_ = n_samples, n_features
        self.components_ = components_[:n_components]
        self.n_components_ = n_components
        self.explained_variance_ = explained_variance_[:n_components]
        self.explained_variance_ratio_ = explained_variance_ratio_[:n_components]
        self.singular_values_ = singular_values_[:n_components]

        return U, S, V

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
        X = check_array(X)
        return np.dot(X, self.components_)
