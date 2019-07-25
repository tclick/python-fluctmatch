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
# The ICA code has been modified from the MNE Python package
# https://martinos.org/mne/stable/index.html
# Gramfort, M. Luessi, E. Larson, D. Engemann, D. Strohmeier,
# C. Brodbeck, R. Goj, M. Jas, T. Brooks, L. Parkkonen, M. Hämäläinen,
# MEG and EEG data analysis with MNE-Python, Frontiers in Neuroscience,
# Volume 7, 2013, ISSN 1662-453X,
import logging
from copy import deepcopy

import numpy as np
from scipy import linalg
from sklearn.decomposition import FastICA
from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.utils.validation import FLOAT_DTYPES
from sklearn.utils.validation import as_float_array
from sklearn.utils.validation import check_array
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.validation import check_random_state

logger: logging.Logger = logging.getLogger(__name__)
logging.captureWarnings(True)


def _infomax(X, n_components: int = None, l_rate: float = None,
             weights: np.ndarray = None, block: float = None,
             w_change: float = 1e-12, anneal_deg: float = 60.,
             anneal_step: float = 0.9, extended: bool = True,
             n_subgauss: int = 1, kurt_size: int = 6000,
             ext_blocks: int = 1, max_iter: int = 200, whiten=True,
             random_state: np.random.RandomState = None, blowup: float = 1e4,
             blowup_fac: float = 0.5, n_small_angle: int = 20,
             use_bias: bool = True,
             verbose: bool = None) -> np.ndarray:
    """Run (extended) Infomax ICA decomposition on raw data.

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_features)
        The whitened data to unmix.
    weights : np.ndarray, shape (n_features, n_features)
        The initialized unmixing matrix.
        Defaults to None, which means the identity matrix is used.
    l_rate : float
        This quantity indicates the relative size of the change in weights.
        Defaults to ``0.01 / log(n_features ** 2)``.

        .. note:: Smaller learning rates will slow down the ICA procedure.

    block : int
        The block size of randomly chosen data segments.
        Defaults to floor(sqrt(n_times / 3.)).
    w_change : float
        The change at which to stop iteration. Defaults to 1e-12.
    anneal_deg : float
        The angle (in degrees) at which the learning rate will be reduced.
        Defaults to 60.0.
    anneal_step : float
        The factor by which the learning rate will be reduced once
        ``anneal_deg`` is exceeded: ``l_rate *= anneal_step.``
        Defaults to 0.9.
    extended : bool
        Whether to use the extended Infomax algorithm or not.
        Defaults to True.
    n_subgauss : int
        The number of subgaussian components. Only considered for extended
        Infomax. Defaults to 1.
    kurt_size : int
        The window size for kurtosis estimation. Only considered for extended
        Infomax. Defaults to 6000.
    ext_blocks : int
        Only considered for extended Infomax. If positive, denotes the number
        of blocks after which to recompute the kurtosis, which is used to
        estimate the signs of the sources. In this case, the number of
        sub-gaussian sources is automatically determined.
        If negative, the number of sub-gaussian sources to be used is fixed
        and equal to n_subgauss. In this case, the kurtosis is not estimated.
        Defaults to 1.
    max_iter : int
        The maximum number of iterations. Defaults to 200.
    random_state : int | np.random.RandomState
        If random_state is an int, use random_state to seed the random number
        generator. If random_state is already a np.random.RandomState instance,
        use random_state as random number generator.
    blowup : float
        The maximum difference allowed between two successive estimations of
        the unmixing matrix. Defaults to 10000.
    blowup_fac : float
        The factor by which the learning rate will be reduced if the difference
        between two successive estimations of the unmixing matrix exceededs
        ``blowup``: ``l_rate *= blowup_fac``. Defaults to 0.5.
    n_small_angle : int | None
        The maximum number of allowed steps in which the angle between two
        successive estimations of the unmixing matrix is less than
        ``anneal_deg``. If None, this parameter is not taken into account to
        stop the iterations. Defaults to 20.
    use_bias : bool
        This quantity indicates if the bias should be computed.
        Defaults to True.
    verbose : bool, str, int, or None
        If not None, override default verbosity level (see :func:`mne.verbose`
        and :ref:`Logging documentation <tut_logging>` for more).

    Returns
    -------
    W : array, shape (n_components, n_components)
        Estimated un-mixing matrix.
        The mixing matrix can be obtained by::

            w = np.dot(W, K.T)
            A = w.T * (w * w.T).I

    References
    ----------
    .. [1] A. J. Bell, T. J. Sejnowski. An information-maximization approach to
           blind separation and blind deconvolution. Neural Computation, 7(6),
           1129-1159, 1995.
    .. [2] T. W. Lee, M. Girolami, T. J. Sejnowski. Independent component
           analysis using an extended infomax algorithm for mixed subgaussian
           and supergaussian sources. Neural Computation, 11(2), 417-441, 1999.
    """
    from scipy.stats import kurtosis
    rng = check_random_state(random_state)

    # define some default parameters
    max_weight = 1e8
    restart_fac = 0.9
    min_l_rate = 1e-10
    degconst = 180.0 / np.pi

    # for extended Infomax
    extmomentum = 0.5
    signsbias = 0.02
    signcount_threshold = 25
    signcount_step = 2

    # check data shape
    n_samples, n_features = X.shape
    n_features_square = np.square(n_features)

    if whiten:
        # Centering the columns (ie the variables)
        X_mean = X.mean(axis=-1)
        X -= X_mean[:, np.newaxis]

        # Whitening and preprocessing by PCA
        u, d, _ = linalg.svd(X, full_matrices=False)

        del _
        K = (u / d).T[:n_components]  # see (6.33) p.140
        del u, d
        X1 = np.dot(K, X)
        # see (13.6) p.267 Here X1 is white and data
        # in X has been projected onto a subspace by PCA
        X1 *= np.sqrt(n_features)
    else:
        # X must be casted to floats to avoid typing issues with numpy
        # 2.0 and the line below
        X1 = as_float_array(X, copy=False)  # copy has been taken care of

    # check input parameters
    # heuristic default - may need adjustment for large or tiny data sets
    if l_rate is None:
        l_rate = 0.01 / np.log(n_features ** 2.0)

    if block is None:
        block = int(np.floor(np.sqrt(n_samples / 3.0)))

    logger.info('Computing%sInfomax ICA' % ' Extended ' if extended else ' ')

    # collect parameters
    nblock = n_samples // block
    lastt = (nblock - 1) * block + 1

    # initialize training
    weights = (np.identity(n_features,
                           dtype=np.float64) if weights is None else weights.T)

    BI = block * np.identity(n_features, dtype=np.float64)
    bias = np.zeros((n_features, 1), dtype=np.float64)
    onesrow = np.ones((1, block), dtype=np.float64)
    startweights = weights.copy()
    oldweights = startweights.copy()
    step = 0
    count_small_angle = 0
    wts_blowup = False
    blockno = 0
    signcount = 0
    initial_ext_blocks = ext_blocks  # save the initial value in case of reset

    # for extended Infomax
    if extended:
        signs = np.ones(n_features)
        signs[:n_subgauss] = -1

        kurt_size = min(kurt_size, n_samples)
        old_kurt = np.zeros(n_features, dtype=np.float64)
        oldsigns = np.zeros(n_features)

    # trainings loop
    olddelta, oldchange = 1., 0.
    while step < max_iter:
        # shuffle data at each step
        permute = random_permutation(n_samples, rng)

        # ICA training block
        # loop across block samples
        for t in range(0, lastt, block):
            u = np.dot(X1[permute[t:t + block], :], weights)
            u += np.dot(bias, onesrow).T

            if extended:
                # extended ICA update
                y = np.tanh(u)
                j = BI - signs[None, :] * np.dot(u.T, y) - np.dot(u.T, u)
                weights += l_rate * np.dot(weights, j)
            if use_bias:
                bias += l_rate * np.reshape(
                    np.sum(y, axis=0, dtype=float) * -2.0, (n_features, 1))
        else:
            # logistic ICA weights update
            y = 1.0 / (1.0 + np.exp(-u))
            j = BI + np.dot(u.T, (1.0 - 2.0 * y))
            weights += l_rate * np.dot(weights, j)

            if use_bias:
                bias += l_rate * np.reshape(np.sum((1.0 - 2.0 * y), axis=0,
                                                   dtype=float),
                                            (n_features, 1))

        # check change limit
        max_weight_val = np.max(np.abs(weights))
        if max_weight_val > max_weight:
            wts_blowup = True

        blockno += 1
        if wts_blowup:
            break

        # ICA kurtosis estimation
        if extended:
            if ext_blocks > 0 and blockno % ext_blocks == 0:
                if kurt_size < n_samples:
                    rp = np.floor(rng.uniform(0, 1, kurt_size) *
                                  (n_samples - 1))
                    tpartact = np.dot(X[rp.astype(int), :], weights).T
                else:
                    tpartact = np.dot(X, weights).T

                # estimate kurtosis
                kurt = kurtosis(tpartact, axis=1, fisher=True)

                if extmomentum != 0:
                    kurt = (extmomentum * old_kurt +
                            (1.0 - extmomentum) * kurt)
                    old_kurt = kurt

                # estimate weighted signs
                signs = np.sign(kurt + signsbias)

                ndiff = (signs - oldsigns != 0).sum()
                if ndiff == 0:
                    signcount += 1
                else:
                    signcount = 0
                oldsigns = signs

                if signcount >= signcount_threshold:
                    ext_blocks = np.fix(ext_blocks * signcount_step)
                    signcount = 0

    # here we continue after the for loop over the ICA training blocks
    # if weights in bounds:
    if not wts_blowup:
        oldwtchange = weights - oldweights
        step += 1
        angledelta = 0.0
        delta = oldwtchange.reshape(1, n_features_square)
        change = np.sum(delta * delta, dtype=np.float64)
        if step > 2:
            angledelta = np.arccos(np.sum(delta * olddelta) /
                                   np.sqrt(change * oldchange))
            angledelta *= degconst

        if verbose:
            logger.info(
                'step %d - lrate %5f, wchange %8.8f, angledelta %4.1f deg'
                % (step, l_rate, change, angledelta))

        # anneal learning rate
        oldweights = weights.copy()
        if angledelta > anneal_deg:
            l_rate *= anneal_step  # anneal learning rate
            # accumulate angledelta until anneal_deg reaches l_rate
            olddelta = delta
            oldchange = change
            count_small_angle = 0  # reset count when angledelta is large
        else:
            if step == 1:  # on first step only
                olddelta = delta  # initialize
                oldchange = change

            if n_small_angle is not None:
                count_small_angle += 1
                if count_small_angle > n_small_angle:
                    max_iter = step

        # apply stopping rule
        if step > 2 and change < w_change:
            step = max_iter
        elif change > blowup:
            l_rate *= blowup_fac

    # restart if weights blow up (for lowering l_rate)
    else:
        step = 0  # start again
        wts_blowup = 0  # re-initialize variables
        blockno = 1
        l_rate *= restart_fac  # with lower learning rate
        weights = startweights.copy()
        oldweights = startweights.copy()
        olddelta = np.zeros((1, n_features_square), dtype=np.float64)
        bias = np.zeros((n_features, 1), dtype=np.float64)

        ext_blocks = initial_ext_blocks

        # for extended Infomax
        if extended:
            signs = np.ones(n_features)
            for k in range(n_subgauss):
                signs[k] = -1
            oldsigns = np.zeros(n_features)

        if l_rate > min_l_rate:
            if verbose:
                logger.info('... lowering learning rate to %g'
                            '\n... re-starting...' % l_rate)
        else:
            raise ValueError('Error in Infomax ICA: unmixing_matrix matrix'
                             'might not be invertible!')

    # prepare return values
    return weights.T


def random_permutation(n_samples, random_state=None):
    """Emulate the randperm matlab function.

    It returns a vector containing a random permutation of the
    integers between 0 and n_samples-1. It returns the same random numbers
    than randperm matlab function whenever the random_state is the same
    as the matlab"s random seed.

    This function is useful for comparing against matlab scripts
    which use the randperm function.

    Note: the randperm(n_samples) matlab function generates a random
    sequence between 1 and n_samples, whereas
    random_permutation(n_samples, random_state) function generates
    a random sequence between 0 and n_samples-1, that is:
    randperm(n_samples) = random_permutation(n_samples, random_state) - 1

    Parameters
    ----------
    n_samples : int
        End point of the sequence to be permuted (excluded, i.e., the end point
        is equal to n_samples-1)
    random_state : int | None
        Random seed for initializing the pseudo-random number generator.

    Returns
    -------
    randperm : ndarray, int
        Randomly permuted sequence between 0 and n-1.
    """
    rng = check_random_state(random_state)
    idx = rng.rand(n_samples)
    randperm = np.argsort(idx)
    return randperm


class ICA(BaseEstimator, TransformerMixin):
    """Signal decomposition using Independent Component Analysis (ICA).

    This object can be used to estimate ICA components and then remove some
    from Raw or Epochs for data exploration or artifact correction.

    Caveat! If supplying a noise covariance, keep track of the projections
    available in the cov or in the raw object. For example, if you are
    interested in EOG or ECG artifacts, EOG and ECG projections should be
    temporally removed before fitting ICA, for example::

        >> projs, raw.info['projs'] = raw.info['projs'], []
        >> ica.fit(raw)
        >> raw.info['projs'] = projs

    .. note:: Methods currently implemented are FastICA (default), Infomax,
              Extended Infomax. Infomax can be quite sensitive to
              differences in floating point arithmetic. Extended Infomax seems
              to be more stable in this respect enhancing reproducibility and
              stability of results.

    .. warning:: ICA is sensitive to low-frequency drifts and therefore
                 requires the data to be high-pass filtered prior to fitting.
                 Typically, a cutoff frequency of 1 Hz is recommended.

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
        The ICA method to use. Defaults to 'extended-infomax'. For reference,
        see [1]_, [2]_, and [3] .
    fit_params : dict | None
        Additional parameters passed to the ICA estimator as specified by
        `method`.
    weights : np.ndarray, shape (n_features, n_features)
        The initialized unmixing matrix.
        Defaults to None, which means the identity matrix is used.
    l_rate : float
        This quantity indicates the relative size of the change in weights.
        Defaults to ``0.01 / log(n_features ** 2)``.

        .. note:: Smaller learning rates will slow down the ICA procedure.
    block : int
        The block size of randomly chosen data segments.
        Defaults to floor(sqrt(n_times / 3.)).
    w_change : float
        The change at which to stop iteration. Defaults to 1e-12.
    anneal_deg : float
        The angle (in degrees) at which the learning rate will be reduced.
        Defaults to 60.0.
    anneal_step : float
        The factor by which the learning rate will be reduced once
        ``anneal_deg`` is exceeded: ``l_rate *= anneal_step.``
        Defaults to 0.9.
    extended : bool
        Whether to use the extended Infomax algorithm or not.
        Defaults to True.
    n_subgauss : int
        The number of subgaussian components. Only considered for extended
        Infomax. Defaults to 1.
    kurt_size : int
        The window size for kurtosis estimation. Only considered for extended
        Infomax. Defaults to 6000.
    ext_blocks : int
        Only considered for extended Infomax. If positive, denotes the number
        of blocks after which to recompute the kurtosis, which is used to
        estimate the signs of the sources. In this case, the number of
        sub-gaussian sources is automatically determined.
        If negative, the number of sub-gaussian sources to be used is fixed
        and equal to n_subgauss. In this case, the kurtosis is not estimated.
        Defaults to 1.
    max_iter : int
        The maximum number of iterations. Defaults to 200.
    random_state : int | np.random.RandomState
        If random_state is an int, use random_state to seed the random number
        generator. If random_state is already a np.random.RandomState instance,
        use random_state as random number generator.
    blowup : float
        The maximum difference allowed between two successive estimations of
        the unmixing matrix. Defaults to 10000.
    blowup_fac : float
        The factor by which the learning rate will be reduced if the difference
        between two successive estimations of the unmixing matrix exceededs
        ``blowup``: ``l_rate *= blowup_fac``. Defaults to 0.5.
    n_small_angle : int | None
        The maximum number of allowed steps in which the angle between two
        successive estimations of the unmixing matrix is less than
        ``anneal_deg``. If None, this parameter is not taken into account to
        stop the iterations. Defaults to 20.
    use_bias : bool
        This quantity indicates if the bias should be computed.
        Defaults to True.

    Attributes
    ----------
    n_components_ : int
        If fit, the actual number of components used for ICA decomposition.
    mixing_ : ndarray, shape (`n_components_`, `n_components_`)
        If fit, the mixing matrix to restore observed data.
    components_ : ndarray, shape (`n_components_`, `n_components_`)
        If fit, the matrix to unmix observed data.
    n_samples_ : int
        The number of samples used on fit.

    Notes
    -----
    Reducing the tolerance speeds up estimation at the cost of consistency of
    the obtained results. It is difficult to directly compare tolerance levels
    between Infomax and Picard, but for Picard and FastICA a good rule of thumb
    is ``tol_fastica = tol_picard ** 2``.

    References
    ----------
    .. [1] Hyvärinen, A., 1999. Fast and robust fixed-point algorithms for
           independent component analysis. IEEE transactions on Neural
           Networks, 10(3), pp.626-634.

    .. [2] Bell, A.J., Sejnowski, T.J., 1995. An information-maximization
           approach to blind separation and blind deconvolution. Neural
           computation, 7(6), pp.1129-1159.

    .. [3] Lee, T.W., Girolami, M., Sejnowski, T.J., 1999. Independent
           component analysis using an extended infomax algorithm for mixed
           subgaussian and supergaussian sources. Neural computation, 11(2),
           pp.417-441.
    """

    def __init__(self, whiten=True, n_components=None, random_state=None,
                 method='fastica', fit_params=None, max_iter=200,
                 verbose=None):
        methods = ('fastica', 'infomax', 'extended-infomax', 'picard')
        if method not in methods:
            raise ValueError('`method` must be "%s". You passed: "%s"' %
                             ('" or "'.join(methods), method))

        if isinstance(n_components, float) and not 0 < n_components <= 1:
            raise ValueError('Selecting ICA components by explained variance '
                             'needs values between 0.0 and 1.0 ')

        self.verbose = verbose
        self.n_components = n_components
        self.random_state = random_state
        self.whiten = whiten

        if fit_params is None:
            fit_params = {}
        fit_params = deepcopy(fit_params)  # avoid side effects
        if "extended" in fit_params:
            raise ValueError("'extended' parameter provided. You should "
                             "rather use method='extended-infomax'.")
        if method == 'fastica':
            update = {
                'algorithm': 'parallel', 'fun': 'logcosh',
                'fun_args': None
            }
            fit_params.update(dict((k, v) for k, v in update.items() if k
                                   not in fit_params))
        elif method == 'infomax':
            fit_params.update({'extended': False})
        elif method == 'extended-infomax':
            fit_params.update({'extended': True})
        if 'max_iter' not in fit_params:
            fit_params['max_iter'] = max_iter
        self.max_iter = max_iter
        self.fit_params = fit_params

        self.method = method

    def __repr__(self):
        """ICA fit information."""
        s = (f"fit ({self.method}): "
             f"{getattr(self, 'n_samples', '')} samples, ")
        s += (
            f"{str(self.n_components)} components"
            if self.n_components is not None
            else "no dimension reduction"
        )

        return f"<ICA  |  {s}>"

    def _reset(self):
        """Aux method."""
        if hasattr(self, "mixing_"):
            del self.components_
            del self.mixing_
            del self.pca_components_
            del self.pca_explained_variance_
            del self.mean_

    def fit(self, data: np.ndarray):
        """Aux function."""
        self._reset()
        self.n_samples, n_features = data.shape
        random_state = check_random_state(self.random_state)

        # take care of ICA
        if self.method == 'fastica':
            ica = FastICA(whiten=self.whiten, random_state=random_state,
                          **self.fit_params)
            ica.fit(data)
            self.components_ = ica.components_
            self.mixing_ = ica.mixing_
        elif self.method in ('infomax', 'extended-infomax'):
            self.components_ = _infomax(data, random_state=random_state,
                                        whiten=self.whiten,
                                        **self.fit_params)[:self.n_components]
            self.mixing_ = linalg.pinv(self.components_)

        return self

    def transform(self, data: np.ndarray, copy: bool = True) -> np.ndarray:
        """Compute sources from data (operates inplace)."""
        check_is_fitted(self, 'mixing_')

        data = check_array(data, copy=copy, dtype=FLOAT_DTYPES)

        # Apply unmixing to low dimension PCA
        sources = np.dot(data, self.components_.T)
        return sources

    def inverse_transform(self, data: np.ndarray,
                          copy: bool = True) -> np.ndarray:
        check_is_fitted(self, 'mixing_')

        data: np.ndarray = check_array(data, copy=copy, dtype=FLOAT_DTYPES)
        data = np.dot(data, self.mixing_.T)
        return data

    def copy(self) -> object:
        return deepcopy(self)
