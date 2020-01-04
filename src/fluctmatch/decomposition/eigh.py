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
#   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS “AS IS”
#   AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
#   IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
#   ARE DISCLAIMED. IN NO EVENT SHALL THE REGENTS OR CONTRIBUTORS BE LIABLE FOR
#   ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
#   DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
#   SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
#   CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
#   OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
#   OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
#   Timothy H. Click, Nixon Raj, and Jhih-Wei Chu.
#   Simulation. Meth Enzymology. 578 (2016), 327-342,
#   Calculation of Enzyme Fluctuograms from All-Atom Molecular Dynamics
#   doi:10.1016/bs.mie.2016.05.024.
#
# ------------------------------------------------------------------------------

from typing import Union

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

    def __init__(self, copy: bool=True):
        self.copy: bool = copy

    def fit(self, X: np.ndarray, y: Union[np.ndarray, None]=None) -> "Eigh":
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

    def transform(self, X: np.ndarray, y: Union[np.ndarray, None]=None) -> np.ndarray:
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
        V: np.ndarray = self._fit(X)
        return V

    def _fit(self, X: np.ndarray) -> np.ndarray:
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
        self.components_: np.ndarray = V.T
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
