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
from scipy import linalg
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import scale, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.utils.validation import check_array, check_is_fitted, FLOAT_DTYPES

from ..decomposition.ica import ICA
from ..decomposition.svd import SVD


class FluctSCA(BaseEstimator, TransformerMixin):
    def __init__(self, n_components: int=None, max_iter: int=1000,
                 whiten: bool=True, stddev: float = 2.0,
                 method: str="extended-infomax"):
        super().__init__()
        self.n_components: int= n_components
        self.max_iter: int= max_iter
        self.stddev: float= stddev
        self.whiten: bool= whiten
        self.method: str= method

    def _randomize(self, X: np.ndarray) -> np.ndarray:
        """Calculates eigenvalues from a random matrix.

        Parameters
        ----------

        """
        X: np.ndarray = check_array(X, copy=True, dtype=FLOAT_DTYPES)
        _, n_windows = X.shape

        mean: np.ndarray = np.mean(X, axis=-1)[:, None]
        mean: np.ndarray = np.tile(mean, (1, n_windows)),
        std: np.ndarray = np.std(X, axis=-1)[:, None]
        std: np.ndarray = np.tile(std, (1, n_windows)),
        positive: bool = np.all(X >= 0.)

        svd: SVD = SVD()
        Lrand: np.ndarray = np.empty((self.max_iter, np.min(X.shape)), dtype=X.dtype)
        for _ in range(self.max_iter):
            Y: np.ndarray = np.random.normal(mean, std)
            if positive:
                Y[Y < 0.] = 0.
            if self.whiten:
                Y: np.ndarray = scale(Y)
            Lrand[_, :] = svd.fit(Y).explained_variance_
        return Lrand

    def _calculate_maxdims(self, X: np.ndarray):
        """Calculate the significant number of eigenvalues.
        """
        X: np.ndarray = check_array(X, accept_sparse=('csr', 'csc'), copy=True,
                                    warn_on_dtype=True, estimator=self,
                                    dtype=FLOAT_DTYPES,
                                    force_all_finite='allow-nan')
        value: float = X[:, 1].mean() + ((self.stddev + 1) * X[:, 1].std())
        self.n_components: int = self.eigenvector_[self.eigenvector_ > value].size

    def fit(self, X: np.ndarray) -> "FluctSCA":
        from sklearn.pipeline import Pipeline

        X: np.ndarray = check_array(X, accept_sparse=('csr', 'csc'), copy=True,
                                    warn_on_dtype=True,
                                    estimator=self, dtype=FLOAT_DTYPES,
                                    force_all_finite='allow-nan')

        svd: SVD = SVD()
        pipeline: Pipeline = (
            make_pipeline(StandardScaler(), svd)
            if self.whiten
            else make_pipeline(svd)
        )
        pipeline.fit(X)
        self.eigenvector_ = svd.explained_variance_

        if self.n_components < 1 or self.n_components is None:
            Lrand: np.ndarray= self._randomize(X)
            self._calculate_maxdims(Lrand)

        return self

    def transform(self, X: np.ndarray, copy: bool=True) -> np.ndarray:
        from sklearn.pipeline import Pipeline

        check_is_fitted(self, "Lsca")
        X: np.ndarray = check_array(X, accept_sparse=('csr', 'csc'), copy=copy,
                        warn_on_dtype=True, estimator=self, dtype=FLOAT_DTYPES,
                        force_all_finite='allow-nan')

        ica: ICA = ICA(n_components=self.n_components, method=self.method,
                       whiten=self.whiten)
        self.sources_: np.ndarray = ica.fit_transform(X)

        # Perform truncated singular value decomposition
        svd: SVD = SVD(n_components=self.n_components)
        pipeline: Pipeline = (
            make_pipeline(StandardScaler(), svd)
            if self.whiten
            else make_pipeline(svd)
        )
        self.U_: np.ndarray = pipeline.fit_transform(X)
        self.components_: np.ndarray = svd.components_
