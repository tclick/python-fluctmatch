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
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import FLOAT_DTYPES, check_array, check_is_fitted


class Center2D(BaseEstimator, TransformerMixin):
    """Center a 2-D matrix by removing the mean from both dimensions

    Parameters
    ----------
    copy : boolean, optional, default True
        If False, try to avoid a copy and do inplace scaling instead.
        This is not guaranteed to always work inplace; e.g. if the data is
        not a NumPy array or scipy.sparse CSR matrix, a copy may still be
        returned.

    with_mean : boolean, True by default
        If True, center the data before scaling.
        This does not work (and will raise an exception) when attempted on
        sparse matrices, because centering them entails building a dense
        matrix which in common use cases is likely to be too large to fit in
        memory.
    """

    def __init__(self, copy: bool = True, with_mean: bool = True):
        self.with_mean: bool = with_mean
        self.copy: bool = copy

    def _reset(self):
        """Reset internal data-dependent state of the scaler, if necessary.

        __init__ parameters are not touched.
        """

        # Checking one attribute is enough, becase they are all set together
        # in partial_fit
        if hasattr(self, "mean_"):
            del self.features_mean_
            del self.samples_mean_
            del self.grand_mean_

    def fit(self, X: np.ndarray, y: np.ndarray = None):
        """Compute the mean to be used for later scaling.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape [n_samples, n_features]
            The data used to compute the mean and standard deviation
            used for later scaling along the features axis.

        y
            Ignored
        """

        # Reset internal state before fitting
        self._reset()
        return self.partial_fit(X)

    def partial_fit(self, X: np.ndarray):
        """Online computation of 2-D mean on X for later scaling.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape [n_samples, n_features]
            The data used to compute the mean and standard deviation
            used for later scaling along the features axis.

        y
            Ignored
        """
        X = check_array(
            X,
            accept_sparse=("csr", "csc"),
            copy=self.copy,
            warn_on_dtype=True,
            estimator=self,
            dtype=FLOAT_DTYPES,
            force_all_finite="allow-nan",
        )

        n_samples, n_features = X.shape
        if self.with_mean:
            self.features_mean_ = X.mean(axis=0)[np.newaxis, :]
            self.samples_mean_ = X.mean(axis=1)[:, np.newaxis]
            self.grand_mean_ = X.mean()
        else:
            self.features_mean_ = np.zeros((1, n_features), dtype=np.float)
            self.samples_mean_ = np.zeros((n_samples, 1), dtype=np.float)
            self.grand_mean_ = 0.0
        self.mean_ = self.features_mean_ + self.samples_mean_ - self.grand_mean_
        return self

    def transform(self, X: np.ndarray):
        """Online computation of 2-D mean on X for later scaling.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape [n_samples, n_features]
            The data used to compute the mean and standard deviation
            used for later scaling along the features axis.

        Returns
        -------
        X_new : array-like, [n_samples, n_features]
            A centered array
        """
        check_is_fitted(self, "mean_")
        X = check_array(
            X,
            accept_sparse=("csr", "csc"),
            copy=self.copy,
            warn_on_dtype=True,
            estimator=self,
            dtype=FLOAT_DTYPES,
            force_all_finite="allow-nan",
        )
        return X - self.features_mean_ - self.samples_mean_ + self.grand_mean_

    def inverse_transform(self, X: np.ndarray, copy: bool = None):
        """Scale back the data to the original representation

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape [n_samples, n_features]
            The data used to compute the mean and standard deviation
            used for later scaling along the features axis.

        copy : boolean, optional, default True
            Copy the input X or not.

        Returns
        -------
        X_new : array-like, [n_samples, n_features]
            Transformed array
        """
        check_is_fitted(self, "mean_")
        X = check_array(
            X,
            accept_sparse=("csr", "csc"),
            copy=copy,
            warn_on_dtype=True,
            estimator=self,
            dtype=FLOAT_DTYPES,
            force_all_finite="allow-nan",
        )
        return X + self.mean_
