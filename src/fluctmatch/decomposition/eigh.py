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
from scipy.sparse import linalg
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
from sklearn.utils import check_array


class Eigh(BaseEstimator, TransformerMixin):
    """Hermitian eigenvalue decomposition.

    This transformer performs eigenvalue decomposition (ED) of
    Hermitian square matrices. The eigenvalues and vectors are
    sorted in descending order of the eigenvalues.

    Parameters
    ----------
    copy : bool (default True)
        If False, data passed to fit are overwritten and running
        fit(X).transform(X) will not yield the expected results,
        use fit_transform(X) instead.

    Attributes
    ----------
    components_ : array, shape (n_components, n_components)
        The transpose of the returned matrix.

    explained_variance_ : array, shape (n_components,)
        Same as the eigenvector.

    explained_variance_ratio_ : array, shape (n_components,)
        Percentage of variance explained by each of the selected components.

    eigenvalues_ : array, shape (n_components,)
        The eigenvalues corresponding to each of the selected components.

    Examples
    --------
    >>> import numpy as np
    >>> from fluctmatch.decomposition.svd import SVD
    >>> X = np.arange(4, dtype=np.float).reshape((2,2))
    >>> eigh = Eigh()
    >>> eigh.fit(X)
    Eigh(copy=True)
    >>> print(eigh.explained_variance_ratio_)  # doctest: +ELLIPSIS
    [-1... 4...]
    >>> print(eigh.singular_values_)  # doctest: +ELLIPSIS
    [-1... 4...]

    Notes
    -----
    SVD suffers from a problem called "sign indeterminacy", which means the
    sign of the ``components_`` and the output from transform depend on the
    algorithm and random state. To work around this, fit instances of this
    class to data once, then keep the instance around to do transformations.
    """

    def __init__(self, copy=True):
        self.copy = copy

    def fit(self, X: np.ndarray, y=None) -> object:
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

    def transform(self, X: np.ndarray, y=None) -> np.ndarray:
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
        V = self._fit(X)
        return V

    def _fit(self, X: np.ndarray):
        """Dispatch to the right submethod depending on the chosen solver."""

        # Raise an error for sparse input.
        # This is more informative than the generic one raised by check_array.
        X: np.ndarray = check_array(X, dtype=[np.float64, np.float32],
                                    ensure_2d=True, copy=self.copy)
        L, V = linalg.eigsh(X, k=X.shape[0])
        idx: np.ndarray = np.argsort(L)[::-1]
        self.eigenvalues_: np.ndarray = L[idx].copy()
        self.explained_variance_: np.ndarray = L[idx].copy()
        total_eigv: float = L.sum()
        self.explained_variance_ratio_: np.ndarray = L / total_eigv
        V: np.ndarray = V[:, idx]
        self.components_ = V.T
        return V

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
        return np.dot(X * self.eigenvalues_, self.components_)
