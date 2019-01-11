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
from sklearn.pipeline import make_pipeline
from sklearn.utils.validation import check_array, check_is_fitted, FLOAT_DTYPES

from ..decomposition.ica import ICA
from ..decomposition.svd import SVD
from .finddims import FindDims


class FluctSCA(BaseEstimator, TransformerMixin):
    def __init__(self, n_components: int=None, max_iter: int=1000,
                 whiten: bool=True, stddev: int= 2., tol: float=0.99,
                 method: str="infomax", svd_solver: str="auto",
                 random_state: RandomState=None):
        super().__init__()
        self.n_components: int= n_components
        self.max_iter: int= max_iter
        self.stddev: int= stddev
        self.whiten: bool= whiten
        self.tol: float= tol
        self.method: str= method
        self.svd_solver: str= svd_solver
        self.random_state: RandomState= random_state

    def fit(self, X: np.ndarray) -> "FluctSCA":
        from sklearn.pipeline import Pipeline

        X: np.ndarray = check_array(X, accept_sparse=('csr', 'csc'), copy=True,
                                    warn_on_dtype=True,
                                    estimator=self, dtype=FLOAT_DTYPES,
                                    force_all_finite='allow-nan')

        if self.n_components is None:
            fd: FindDims = FindDims(whiten=self.whiten, max_iter=self.max_iter,
                                    stddev=self.stddev, tol=self.tol,
                                    algorithm=self.svd_solver,
                                    random_state=self.random_state)
            self.n_components_ = fd.fit_transform(X)
        else:
            self.n_components_ = self.n_components

        # Perform singular value decomposition
        svd: SVD = SVD(n_components=self.n_components_)
        pipeline: Pipeline = (
            make_pipeline(StandardScaler(), svd)
            if self.whiten
            else make_pipeline(svd)
        )
        pipeline.fit(X)
        self.Vfeatures_: np.ndarray = svd.components_.T
        self.singular_values_: np.ndarray = svd.singular_values_.copy()

        return self

    def transform(self, X: np.ndarray, copy: bool=True) -> np.ndarray:
        from sklearn.pipeline import Pipeline

        check_is_fitted(self, "Vfeatures_")
        X: np.ndarray = check_array(X, accept_sparse=('csr', 'csc'), copy=copy,
                        warn_on_dtype=True, estimator=self, dtype=FLOAT_DTYPES,
                        force_all_finite='allow-nan')

        ica: ICA = ICA(n_components=self.n_components, method=self.method,
                       whiten=self.whiten)

        Usamples_: np.ndarray= X.dot(self.Vfeatures_ / self.singular_values_)

        # Perform ICA on SVD feature covariance matrix
        Vfica: np.ndarray = ica.fit_transform(self.Vfeatures_)
        # Project feature matrix onto sample covariance matrix.
        self.Ufica_: np.ndarray = np.dot(ica.mixing_, Usamples_.T).T
        # Perform ICA on sample covariance matrix
        self.Usica_: np.ndarray = ica.fit_transform(Usamples_)

        return Vfica
