#  python-fluctmatch -
#  Copyright (c) 2019 Timothy H. Click, Ph.D.
#
#  All rights reserved.
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:
#
#  Redistributions of source code must retain the above copyright notice, this
#  list of conditions and the following disclaimer.
#
#  Redistributions in binary form must reproduce the above copyright notice,
#  this list of conditions and the following disclaimer in the documentation
#  and/or other materials provided with the distribution.
#
#  Neither the name of the author nor the names of its contributors may be used
#  to endorse or promote products derived from this software without specific
#  prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS “AS IS”
#  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
#  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
#  ARE DISCLAIMED. IN NO EVENT SHALL THE REGENTS OR CONTRIBUTORS BE LIABLE FOR
#  ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
#  DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
#  SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
#  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
#  OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
#  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
"""Performs independent principal component analysis (see [1].

References
----------
.. [1]  Yao, F.; Coquery, J.; Le Cao, K.-A. 2012. Independent Principal
        Component Analysis for biologically meaningful dimension reduction of
        large biological data sets. BMC Bioinformatics 13 (1).
"""

from typing import Union

import numpy as np
from scipy import linalg, stats
from sklearn.decomposition.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.utils.validation import check_array, check_is_fitted

from .ica import ICA


class IPCA(BaseEstimator, TransformerMixin):
    """Signal decomposition using Independent Principal Component Analysis (IPCA).

    This object can be used to estimate ICA components and then remove some
    from Raw or Epochs for data exploration or artifact correction.

    .. note:: Methods currently implemented are FastICA (default), Infomax,
              Extended Infomax. Infomax can be quite sensitive to differences in
              floating point arithmetic. Extended Infomax seems to be more
              stable in this respect enhancing reproducibility and stability of
              results.

    Parameters
    ----------
    n_components : int | float | None
        Number of components to extract. If None no dimension reduction
        is performed.
    whiten : boolean, optional
        If whiten is false, the data is already considered to be
        whitened, and no whitening is performed.
    random_state : None | int | instance of np.random.RandomState
        Random state to initialize ICA estimation for reproducible results.
    method : {'fastica', 'infomax', 'extended-infomax'}
        The ICA method to use. Defaults to 'fastica'. For reference, see [2]_,
        [3]_, and [4] .
    max_iter : int
        The maximum number of iterations. Defaults to 200.
    random_state : int | np.random.RandomState
        If random_state is an int, use random_state to seed the random number
        generator. If random_state is already a np.random.RandomState instance,
        use random_state as random number generator.

    Attributes
    ----------
    components_ : ndarray, shape (`n_samples`, `n_components`)
        If fit, the matrix to unmix observed data.

    References
    ----------
    .. [2] Hyvärinen, A., 1999. Fast and robust fixed-point algorithms for
           independent component analysis. IEEE transactions on Neural
           Networks, 10(3), pp.626-634.

    .. [3] Bell, A.J., Sejnowski, T.J., 1995. An information-maximization
           approach to blind separation and blind deconvolution. Neural
           computation, 7(6), pp.1129-1159.

    .. [4] Lee, T.W., Girolami, M., Sejnowski, T.J., 1999. Independent
           component analysis using an extended infomax algorithm for mixed
           subgaussian and supergaussian sources. Neural computation, 11(2),
           pp.417-441.
    """
    def __init__(self, n_components: Union[int, float, str]=None,
                 whiten: bool=True, max_iter: int=1000, copy=True,
                 method: str= "fastica",
                 random_state: np.random.RandomState=None):
        self.n_components: Union[int, float, str] = n_components
        self.whiten: bool = whiten
        self.max_iter: int = max_iter
        self.copy: bool = copy
        self.method: str = method
        self.random_state: np.random.RandomState = random_state

    def fit(self, X: np.ndarray, y=None) -> "IPCA":
        scale: StandardScaler = StandardScaler(with_std=self.whiten)
        pca: PCA = PCA(n_components=self.n_components, svd_solver="full",
                       copy=self.copy)
        pca_pipeline: Pipeline = make_pipeline(scale, pca)
        self.pca_projection_: np.ndarray = pca_pipeline.fit_transform(X)
        self.components_: np.ndarray = pca.components_
        self.singular_values_: np.ndarray = pca.singular_values_
        self.explained_variance_: np.ndarray = pca.explained_variance_
        self.explained_variance_ratio_: np.ndarray = pca.explained_variance_ratio_
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self, "components_")

        X: np.ndarray = check_array(X, copy=self.copy)
        X = StandardScaler().fit_transform(X)
        scale: StandardScaler = StandardScaler()
        ica: ICA = ICA(whiten=False, method=self.method, max_iter=self.max_iter,
                       random_state=self.random_state)
        ica_pipeline: Pipeline = make_pipeline(scale, ica)
        S: np.ndarray = ica_pipeline.fit_transform(self.components_.T)

        # Sort signals by kurtosis and reduce dimensions.
        kurtosis: np.ndarray = stats.kurtosis(S)
        idx: np.ndarray = np.argsort(-kurtosis)
        self.kurtosis_: np.ndarray = kurtosis[idx]
        S: np.ndarray = S[:, idx][:, np.where(np.abs(self.kurtosis_) >= 1.)[0]]
        S /= linalg.norm(S, ord=2)
        self.signal_ = S.copy()
        return S
