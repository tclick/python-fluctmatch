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
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.utils.validation import (check_array, FLOAT_DTYPES,
                                      check_random_state, check_is_fitted)

from ..decomposition.svd import SVD


class FindDims(BaseEstimator, TransformerMixin):
    """Find number of dimensions for dimension reduction.

    """
    def __init__(self, whiten: bool=True, max_iter: int=100,
                 stddev: int=2, random_state: RandomState=None,
                 tol: float=0.99):
        self.whiten: bool=whiten
        self.max_iter: int= max_iter
        self.stddev: int= stddev
        self.random_state: RandomState= random_state
        self.tol: float= tol

    def fit(self, X: np.ndarray) -> "FindDims":
        X: np.ndarray = check_array(X, copy=True, dtype=FLOAT_DTYPES)
        random_state = check_random_state(self.random_state)
        Xt = X.T
        n_samples, n_features = X.shape

        scaler: StandardScaler = StandardScaler()
        svd: SVD = SVD()
        pipeline: Pipeline = (
            make_pipeline(scaler, svd)
            if self.whiten
            else make_pipeline(svd)
        )

        scaler.fit(Xt)
        self.mean_: np.ndarray = np.tile(scaler.mean_[None, :], (n_features, 1)).T
        self.std_: np.ndarray = np.tile(scaler.var_[None, :], (n_features, 1)).T
        self.positive_: bool = np.all(X >= 0.)

        self.random_: np.ndarray = np.empty((self.max_iter, np.min(X.shape)),
                                            dtype=X.dtype)
        for _ in range(self.max_iter):
            Y: np.ndarray = random_state.normal(self.mean_, self.std_)
            if self.positive_:
                Y[Y < 0.] = 0.
            pipeline.fit(Y)
            self.random_[_, :] = svd.singular_values_.copy()
        return self

    def transform(self, X: np.ndarray) -> int:
        check_is_fitted(self, ["random_"])

        scaler: StandardScaler = StandardScaler()
        svd: SVD = SVD()
        pipeline: Pipeline = (
            make_pipeline(scaler, svd)
            if self.whiten
            else make_pipeline(svd)
        )
        pipeline.fit(X)

        if self.whiten:
            self.eigenvector_ = eigenvector_ = svd.singular_values_
            mean: np.ndarray = self.random_.mean(axis=1)[1]
            std: np.ndarray = self.random_.std(axis=1)[1]
            value: float = mean + ((self.stddev + 1) * std)
            n_components: int = eigenvector_[eigenvector_ > value].size
        else:
            self.eigenvector_ = explained_ratio = svd.explained_variance_ratio_.cumsum()
            n_components: int = explained_ratio[explained_ratio <= self.tol].size

        return n_components
