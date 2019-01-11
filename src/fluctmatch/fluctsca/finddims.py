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
from numpy.random import RandomState
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import scale, StandardScaler
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.utils.validation import check_array, check_is_fitted, FLOAT_DTYPES

from ..decomposition.svd import SVD


class FindDims(BaseEstimator, TransformerMixin):
    """Find number of dimensions for dimension reduction.

    """
    def __init__(self, whiten: bool=True, max_iter: int=100,
                 stddev: int=2, random_state: RandomState=None,
                 tol=0.99, algorithm="auto"):
        self.whiten: bool=whiten
        self.max_iter: int= max_iter
        self.stddev: int= stddev
        self.random_state = random_state
        self.tol = tol
        self.algorithm = algorithm

    def fit(self, X: np.ndarray) -> "FindDims":
        X: np.ndarray = check_array(X, copy=True, dtype=FLOAT_DTYPES)
        n_samples, n_features = X.shape

        scaler: StandardScaler = StandardScaler()
        svd: SVD = SVD(random_state=self.random_state,
                       iterated_power=self.max_iter,
                       algorithm=self.algorithm)
        pipeline: Pipeline = (
            make_pipeline(scaler, svd)
            if self.whiten
            else make_pipeline(svd)
        )

        scaler.fit(X)
        self.mean_: np.ndarray = np.tile(scaler.mean_[None, :], (n_samples, 1))
        self.std_: np.ndarray = np.tile(scaler.var_[None, :], (n_samples, 1))
        self.positive_: bool = np.all(X >= 0.)

        self.random_: np.ndarray = np.empty((self.max_iter, n_features),
                                            dtype=X.dtype)
        for _ in range(self.max_iter):
            Y: np.ndarray = np.random.normal(self.mean_, self.std_)
            if self.positive_:
                Y[Y < 0.] = 0.
            pipeline.fit(Y)
            self.random_[_, :] = svd.explained_variance_.copy()
        return self

    def transform(self, X: np.ndarray) -> int:
        scaler: StandardScaler = StandardScaler()
        svd: SVD = SVD(random_state=self.random_state,
                       iterated_power=self.max_iter,
                       algorithm=self.algorithm)
        pipeline: Pipeline = (
            make_pipeline(scaler, svd)
            if self.whiten
            else make_pipeline(svd)
        )
        pipeline.fit(X)

        self.eigenvector_: np.ndarray = svd.explained_variance_.copy()
        if self.whiten:
            eigenvector_: np.ndarray = svd.explained_variance_
            mean: np.ndarray = self.random_.mean(axis=1)[1]
            std: np.ndarray = self.random_.std(axis=1)[1]
            value: float = mean + ((self.stddev + 1) * std)
            n_components: int = eigenvector_[eigenvector_ > value].size
        else:
            explained_ratio: np.ndarray = svd.explained_variance_ratio_.cumsum()
            n_components: int = explained_ratio[explained_ratio <= self.tol].size

        self.n_components: int = n_components
        return n_components
