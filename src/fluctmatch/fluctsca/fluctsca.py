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
from scipy import linalg
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
from sklearn.utils.validation import FLOAT_DTYPES, check_array, check_is_fitted

from fluctmatch.decomposition.ica import ICA

from ..libs.center import Center2D


class FluctSCA(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        n_components: int = None,
        max_iter: int = 1000,
        whiten: bool = True,
        stddev: float = 2.0,
        method: str = "extended-infomax",
    ):
        super().__init__()
        self.n_components_: int = n_components
        self.max_iter: int = max_iter
        self.stddev: float = stddev
        self.whiten = whiten
        self.method: str = method

    def _randomize(self, X: np.ndarray) -> np.ndarray:
        """Calculates eigenvalues from a random matrix.

        Parameters
        ----------

        """
        X: np.ndarray = check_array(X, copy=True, dtype=FLOAT_DTYPES)
        _, n_windows = X.shape

        mean: np.ndarray = np.mean(X, axis=-1)[:, None]
        mean: np.ndarray = np.tile(mean, (1, n_windows))
        std: np.ndarray = np.std(X, axis=-1)[:, None]
        std: np.ndarray = np.tile(std, (1, n_windows))
        positive = np.all(X >= 0.0)

        Lrand = np.empty((self.max_iter, np.min(X.shape)), dtype=X.dtype)
        for _ in range(self.max_iter):
            Y = np.random.normal(mean, std)
            if positive:
                Y[Y < 0.0] = 0.0
            if self.whiten:
                Y = Center2D().fit_transform(Y)
            Lrand[_, :] = linalg.svdvals(Y).copy()
        return Lrand

    def _calculate_maxdims(self, X: np.ndarray):
        """Calculate the significant number of eigenvalues.
        """
        X = check_array(
            X,
            accept_sparse=("csr", "csc"),
            copy=True,
            estimator=self,
            dtype=FLOAT_DTYPES,
            force_all_finite="allow-nan",
        )
        value: float = X[:, 1].mean() + ((self.stddev + 1) * X[:, 1].std())
        self.n_components_: int = self.singular_values_[
            self.singular_values_ > value
        ].size

    def fit(self, X: np.ndarray) -> "FluctSCA":
        X = check_array(
            X,
            accept_sparse=("csr", "csc"),
            copy=True,
            estimator=self,
            dtype=FLOAT_DTYPES,
            force_all_finite="allow-nan",
        )

        if self.whiten:
            X = Center2D().fit_transform(X)
        self.singular_values_ = linalg.svdvals(X).copy()
        if self.n_components_ < 1 or self.n_components_ is None:
            self._randomize(X)
            self._calculate_maxdims()

        return self

    def transform(self, X: np.ndarray, copy: bool = True):
        check_is_fitted(self, "Lsca")
        X = check_array(
            X,
            accept_sparse=("csr", "csc"),
            copy=copy,
            warn_on_dtype=True,
            estimator=self,
            dtype=FLOAT_DTYPES,
            force_all_finite="allow-nan",
        )

        ica = ICA(
            n_components=self.n_components_, method=self.method, whiten=self.whiten,
        )
        self.sources_ = ica.fit_transform(X)

        # Perform truncated singular value decomposition
        truncated = TruncatedSVD(n_components=self.n_components_, n_iter=self.max_iter)
        pipeline = make_pipeline(Center2D(), truncated)
        self.U_ = pipeline.fit_transform(X)
