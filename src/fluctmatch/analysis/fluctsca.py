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

from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy import linalg
from scipy.stats import scoreatpercentile, t
from sklearn.utils.extmath import svd_flip
from sklearn.utils.validation import FLOAT_DTYPES, check_array


def fluct_stats(X: np.ndarray) -> dict:
    """

    Parameters
    ----------
    X : array-like, (n_residues, n_windows)
        Table of coupling strengths

    Returns
    -------
    D : dict
        Contains n_residues x n_windows arrays of the mean and standard
        deviations for residues across all windows. 'positive' is True
        if the original matrix was >= 0.
    """
    X: np.ndarray = check_array(X, copy=True, dtype=FLOAT_DTYPES)
    _, n_windows = X.shape

    mean_: np.ndarray = np.mean(X, axis=-1)[:, None]
    std_: np.ndarray = np.std(X, axis=-1)[:, None]

    D: dict = dict(
        mean=np.tile(mean_, (1, n_windows)),
        std=np.tile(std_, (1, n_windows)),
        positive=np.all(X >= 0.0),
    )
    return D


def randomize(X: np.ndarray, n_trials: int = 100) -> np.ndarray:
    """Calculates eigenvalues from a random matrix.

    Parameters
    ----------
    n_trials : int, optional
        Number of trials for eigenvalues
    positive : bool, optional
        If True,

    Returns
    -------
        Array of eigenvalues
    """
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()

    X: np.ndarray = check_array(X, copy=True, dtype=FLOAT_DTYPES)
    n_samples, _ = X.shape
    scaler.fit(X)
    positive = np.all(X >= 0.0)
    mean: np.ndarray = np.tile(scaler.mean_, (n_samples, 1))
    std: np.ndarray = np.tile(scaler.var_, (n_samples, 1))

    Lrand = []
    # pca = PCA(whiten=True, svd_solver='full')
    for _ in range(n_trials):
        Y = np.random.normal(mean, std)
        if positive:
            Y[Y < 0.0] = 0.0
        corr = get_correlation(Y)
        L, _ = eigenVect(corr)
        Lrand.append(L)

    return np.array(Lrand)


def svd(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculate the singular value decomposition with an appropriate sign flip.

    Parameters
    ----------
    X : np.ndarray, [n_samples, n_features]
        Matrix to decompose.

    Returns
    -------
    U : ndarray
        Unitary matrix having left singular vectors as columns.
        Of shape ``(M, M)`` or ``(M, K)``, depending on `full_matrices`.
    s : ndarray
        The singular values, sorted in non-increasing order.
        Of shape (K,), with ``K = min(M, N)``.
    Vh : ndarray
        Unitary matrix having right singular vectors as rows.
        Of shape ``(N, N)`` or ``(K, N)`` depending on `full_matrices`.
    """
    X: np.ndarray = check_array(X, copy=True, dtype=FLOAT_DTYPES)

    U, W, Vt = linalg.svd(X, full_matrices=False)
    U, Vt = svd_flip(U, Vt)
    return U, W, Vt


def get_correlation(X: np.ndarray) -> np.ndarray:
    X: np.ndarray = check_array(X, copy=True, dtype=FLOAT_DTYPES)
    corr: np.ndarray = np.corrcoef(X, rowvar=False)
    corr /= np.diag(corr)
    return corr


def eigenVect(M: np.ndarray):
    """ Return the eigenvectors and eigenvalues, ordered by decreasing values of
    the eigenvalues, for a real symmetric matrix M. The sign of the eigenvectors
    is fixed so that the mean of its components is non-negative.

    :Example:
       >>> eigenVectors, eigenValues = eigenVect(M)
    """
    eigenValues, eigenVectors = linalg.eigh(M)
    idx: np.ndarray = eigenValues.argsort()[::-1]
    eigenValues: np.ndarray = eigenValues[idx]
    eigenVectors: np.ndarray = eigenVectors[:, idx]
    for k in range(eigenVectors.shape[1]):
        if np.sign(np.mean(eigenVectors[:, k])) != 0:
            eigenVectors[:, k] *= np.sign(np.mean(eigenVectors[:, k]))
    return eigenValues, eigenVectors


def correlate(Usca: np.ndarray, Lsca: np.ndarray, kmax: int = 6) -> List[np.ndarray]:
    """Calculate the correlation matrix of *Usca* with *Lsca* eigenvalues.

    Parameters
    ----------
    Usca : :class:`numpy.array`
        Eigenvector
    Lsca : :class:`numpy.array`
        Eigenvalue
    kmax : int, optional
        Number of eigenvectors/eigenvalues to use

    Returns
    -------
    Correlation matrix
    """
    S: np.ndarray = np.power(Lsca, 2)
    Ucorr: List[np.ndarray] = [
        np.outer(Usca[:, _].dot(S[_]), Usca.T[_]) for _ in range(kmax)
    ]
    return Ucorr


def chooseKpos(Lsca: np.ndarray, Lrand: np.ndarray, stddev: float = 2.0) -> int:
    """Calculate the significant number of eigenvalues.

    Parameters
    ----------
    Lsca : :class:`numpy.array`
        Eigenvector from coupling strengths
    Lrand : :class:`numpy.array`
        Matrix of eigenvalues from randomized coupling strengths
    stddev : int, optional
        Number of standard deviations to use

    Returns
    -------
    Number of significant eigenvalues
    """
    value: float = Lrand[:, 1].mean() + ((stddev + 1) * Lrand[:, 1].std())
    return Lsca[Lsca > value].size


def figUnits(
    v1,
    v2,
    v3,
    units,
    filename,
    fig_path=Path.cwd(),
    marker="o",
    dotsize=9,
    notinunits=1,
):
    """ 3d scatter plot specified by 'units', which must be a list of elements
    in the class Unit_. See figColors_ for the color code. Admissible color
    codes are in [0 1] (light/dark gray can also be obtained by using -1/+1).
    For instance: 0->red, 1/3->green, 2/3-> blue.

    .. _Unit: scaTools.html#scaTools.Unit
    .. _figColors: scaTools.html#scaTools.figColors

    **Arguments:**
       -  `v1` = xvals
       -  `v2` = yvals
       -  `units` = list of elements in units

    **Keyword Arguments**
       -  `marker` = plot marker symbol
       -  `dotsize` = specify marker/dotsize
       -  `notinunits` = if set to 1 : the elements not in a unit are
       represented in white, if set to 0 these elements are not represented, if
       set to [w1,w2] : elements with coordinates w1,w2 are represented in white
       in the background.

    :Example:
     >>> figUnits(v1, v2, units, marker='o', gradcol=0, dotsize=9, notinunits=1)

     """
    import colorsys

    # Plot all items in white:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.elev, ax.azim = 30.0, 60.0
    ax.axes

    if notinunits == 1:
        ax.plot(
            v1,
            v2,
            v3,
            marker,
            markersize=dotsize,
            markerfacecolor="w",
            markeredgecolor="k",
        )
    elif len(notinunits) == 3:
        ax.plot(
            notinunits[0],
            notinunits[1],
            notinunits[2],
            marker,
            markersize=dotsize,
            markerfacecolor="w",
            markeredgecolor="k",
        )

    # Plot items in the units with colors:
    for u in units:
        items_list = list(u.items)
        if u.col >= 0 and u.col < 1:
            bgr = colorsys.hsv_to_rgb(u.col, 1, 1)
        if u.col == 1:
            bgr = [0.3, 0.3, 0.3]
        if u.col < 0:
            bgr = [0.7, 0.7, 0.7]
        ax.plot(
            v1[np.ix_(items_list)],
            v2[np.ix_(items_list)],
            v3[np.ix_(items_list)],
            marker,
            markersize=dotsize,
            markerfacecolor=bgr,
            markeredgecolor="k",
        )

    ax.set_xlabel("IC{:d}".format(1))
    ax.set_ylabel("IC{:d}".format(2))
    ax.set_zlabel("IC{:d}".format(3))
    fig.tight_layout()
    fig.savefig(Path(fig_path) / "svd_ica" / filename, dpi=600)


# From pySCA 6.0
class Unit:
    """ A class for units (sectors, sequence families, etc.)

        **Attributes:**
            -  `name`  = string describing the unit (ex: "firmicutes")
            -  `items` = set of member items (ex: indices for all firmicutes
            sequences in an alignment) -  `col`   = color code associated to the
            unit (for plotting) -  `vect`  = an additional vector describing the
            member items (ex: a list of sequence weights)
    """

    def __init__(self):
        self.name: str = ""
        self.items: set = set()
        self.col: float = 0
        self.vect: np.ndarray = 0


def icList(Vpica: np.ndarray, kpos: int, Csca: np.ndarray, p_cut: float = 0.95):
    """ Produces a list of positions contributing to each independent component.

    a defined statistical cutoff (p_cut, the cutoff on the CDF of the
    t-distribution fit to the histogram of each IC).  Any position above the
    cutoff on more than one IC are assigned to one IC based on which group of
    positions to which it shows a higher degree of coevolution. Additionally
    returns the numeric value of the cutoff for each IC, and the pdf fit, which
    can be used for plotting/evaluation. icList, icsize, sortedpos, cutoff,
    pd = icList(Vsca,Lsca,Lrand)

    Parameters
    ----------
    Vpica : :class:`numpy.ndarray`
        Independent component analysis vectors
    kpos : int
        Number of sectors to select
    Csca : :class:`numpy.ndarray`
        Correlation matrix
    p_cut : float
        PDF cutoff

    Returns
    -------
    List of positions contributing to each independent component
    """
    # do the PDF/CDF fit, and assign cutoffs
    Npos, _ = Vpica.shape
    cutoff: list = []
    scaled_pdf: list = []
    all_fits: list = []
    for k in range(kpos):
        pd: np.ndarray = t.fit(Vpica[:, k])
        all_fits.append(pd)
        iqr: float = (
            scoreatpercentile(Vpica[:, k], 75) - scoreatpercentile(Vpica[:, k], 25)
        )
        binwidth: float = 2 * iqr * np.power(Npos, -0.33)
        nbins: int = np.round(
            (Vpica[:, k].max() - Vpica[:, k].min()) / binwidth
        ).astype(np.int)
        hist, bin_edges = np.histogram(Vpica[:, k], nbins)
        x_dist: np.ndarray = np.linspace(bin_edges.min(), bin_edges.max(), num=100)
        area_hist: np.ndarray = Npos * (bin_edges[2] - bin_edges[1])
        scaled_pdf.append(area_hist * t.pdf(x_dist, *pd))
        cd: np.ndarray = t.cdf(x_dist, *pd)
        tmp: np.ndarray = scaled_pdf[k].argmax()
        if np.abs(Vpica[:, k].max()) > np.abs(Vpica[:, k].min()):
            tail: np.ndarray = cd[tmp:]
        else:
            cd: np.ndarray = 1 - cd
            tail: np.ndarray = cd[:tmp]
        diff: np.ndarray = np.abs(tail - p_cut)
        x_pos: np.ndarray = diff.argmin()
        cutoff.append(x_dist[x_pos + tmp])

    # select the positions with significant contributions to each IC
    ic_init: list = [np.where(Vpica[:, k] > cutoff[k])[0] for k in range(kpos)]

    # construct the sorted, non-redundant iclist
    sortedpos = []
    icsize = []
    ics = []
    Csca_nodiag: np.ndarray = Csca.copy()
    np.fill_diagonal(Csca_nodiag, 0.0)
    for k in range(kpos):
        icpos_tmp: list = list(ic_init[k])
        for kprime in (kp for kp in range(kpos) if (kp != k)):
            tmp = [v for v in icpos_tmp if v in ic_init[kprime]]
            for i in tmp:
                remsec = np.linalg.norm(Csca_nodiag[i, ic_init[k]]) < np.linalg.norm(
                    Csca_nodiag[i, ic_init[kprime]]
                )
                if remsec:
                    icpos_tmp.remove(i)
        sortedpos += sorted(icpos_tmp, key=lambda i: -Vpica[i, k])
        icsize.append(len(icpos_tmp))
        s: Unit = Unit()
        s.items: np.ndarray = sorted(icpos_tmp, key=lambda i: -Vpica[i, k])
        s.col: float = k / kpos
        s.vect: np.ndarray = -Vpica[s.items, k]
        ics.append(s)
    return ics, icsize, sortedpos, cutoff, scaled_pdf, all_fits


def basicICA(x, r, Niter):
    """Basic ICA algorithm

    Based on work by Bell & Sejnowski (infomax). The input data should
    preferentially be sphered, i.e., x.T.dot(x) = 1

    Parameters
    ----------
    x : :class:`numpy.ndarray`
        LxM input matrix where L = # features and M = # samples
    r : float
        learning rate / relaxation parameter (e.g. r=.0001)
    Niter : int
        number of iterations (e.g. 1000)

    Returns
    -------
    w : :class:`numpy.ndarray`
        unmixing matrix
    change : :class:`numpy.ndarray`
        record of incremental changes during the iterations.

    Note
    ----
    r and Niter should be adjusted to achieve convergence, which should be
    assessed by visualizing 'change' with plot(range(iter) ,change)

    Example
    -------
    >>> [w, change] = basicICA(x, r, Niter)
    """
    [L, M] = x.shape
    w = np.eye(L)
    change = list()
    for _ in range(Niter):
        w_old = np.copy(w)
        u = w.dot(x)
        w += r * (M * np.eye(L) + (1 - 2 * (1.0 / (1 + np.exp(-u)))).dot(u.T)).dot(w)
        delta = (w - w_old).ravel()
        change.append(delta.dot(delta.T))
    return [w, change]


def rotICA(V, kmax=6, learnrate=0.0001, iterations=10000):
    """ ICA rotation

    Uses basicICA with default parameters and normalization of outputs.

    Example
    -------
    >>> Vica, W = rotICA(V, kmax=6, learnrate=.0001, iterations=10000)
    """
    V1 = V[:, :kmax].T
    [W, changes_s] = basicICA(V1, learnrate, iterations)
    Vica = (W.dot(V1)).T
    for n in range(kmax):
        imax = abs(Vica[:, n]).argmax()
        Vica[:, n] = np.sign(Vica[imax, n]) * Vica[:, n] / np.linalg.norm(Vica[:, n])
    return Vica, W
