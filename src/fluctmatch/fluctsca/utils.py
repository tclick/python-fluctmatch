# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding: utf-8 -*-
# vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4
#
# fluctmatch --- https://github.com/tclick/python-fluctmatch
# Copyright (c) 2013-2017 The fluctmatch Development Team and contributors
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
import multiprocessing as mp
import os
from os import path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy import linalg
from scipy.stats import (scoreatpercentile, t)
from sklearn.decomposition import PCA
from sklearn.utils.extmath import svd_flip
from sklearn.utils.validation import check_array, FLOAT_DTYPES


def figUnits(v1, v2, v3, units, filename, fig_path=os.getcwd(), marker='o',
             dotsize=9, notinunits=1):
    ''' 3d scatter plot specified by 'units', which must be a list of elements
    in the class Unit_. See figColors_ for the color code. Admissible color codes are in [0 1]
    (light/dark gray can also be obtained by using -1/+1).
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
       -  `notinunits` = if set to 1 : the elements not in a unit are represented in white, if set to 0
                         these elements are not represented, if set to [w1,w2] : elements with coordinates
                         w1,w2 are represented in white in the background.

    :Example:
      >>> figUnits(v1, v2, units, marker='o', gradcol=0, dotsize=9, notinunits=1)

     '''
    import colorsys

    # Plot all items in white:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.elev, ax.azim = 30., 60.
    ax.axes

    if notinunits == 1:
        ax.plot(
            v1,
            v2,
            v3,
            marker,
            markersize=dotsize,
            markerfacecolor='w',
            markeredgecolor='k')
    elif len(notinunits) == 3:
        ax.plot(
            notinunits[0],
            notinunits[1],
            notinunits[2],
            marker,
            markersize=dotsize,
            markerfacecolor='w',
            markeredgecolor='k')

    # Plot items in the units with colors:
    for u in units:
        items_list = list(u.items)
        if u.col >= 0 and u.col < 1:
            bgr = colorsys.hsv_to_rgb(u.col, 1, 1)
        if u.col == 1:
            bgr = [.3, .3, .3]
        if u.col < 0:
            bgr = [.7, .7, .7]
        ax.plot(
            v1[np.ix_(items_list)],
            v2[np.ix_(items_list)],
            v3[np.ix_(items_list)],
            marker,
            markersize=dotsize,
            markerfacecolor=bgr,
            markeredgecolor='k')

    ax.set_xlabel('IC{:d}'.format(1))
    ax.set_ylabel('IC{:d}'.format(2))
    ax.set_zlabel('IC{:d}'.format(3))
    fig.tight_layout()
    fig.savefig(path.join(fig_path, 'svd_ica', filename), dpi=600)


# From pySCA 6.0
class Unit:
    """ A class for units (sectors, sequence families, etc.)

        **Attributes:**
            -  `name`  = string describing the unit (ex: "firmicutes")
            -  `items` = set of member items (ex: indices for all firmicutes sequences in an alignment)
            -  `col`   = color code associated to the unit (for plotting)
            -  `vect`  = an additional vector describing the member items (ex: a list of sequence weights)

    """

    def __init__(self):
        self.name: str = ""
        self.items: set = set()
        self.col: float = 0
        self.vect: np.ndarray = 0


def icList(Vpica: np.ndarray, kpos: int, Csca: np.ndarray, p_cut: float=0.95):
    """ Produces a list of positions contributing to each independent component (IC) above
    a defined statistical cutoff (p_cut, the cutoff on the CDF of the t-distribution
    fit to the histogram of each IC).  Any position above the cutoff on more than one IC
    are assigned to one IC based on which group of positions to which it shows a higher
    degree of coevolution. Additionally returns the numeric value of the cutoff for each IC, and the
    pdf fit, which can be used for plotting/evaluation.
    icList, icsize, sortedpos, cutoff, pd  = icList(Vsca,Lsca,Lrand) """
    #do the PDF/CDF fit, and assign cutoffs
    Npos, _ = Vpica.shape
    cutoff: list = []
    scaled_pdf: list = []

    all_fits: list = [t.fit(Vpica[:, _]) for _ in range(kpos)]
    iqr: np.ndarray = (scoreatpercentile(Vpica, 75, axis=0) -
                       scoreatpercentile(Vpica, 25, axis=0))
    binwidth: np.ndarray = 2 * iqr * np.power(Npos, -0.33)
    nbins: np.ndarray = np.round((Vpica.max(axis=0) - Vpica.min(axis=0)) /
                                 binwidth).astype(np.int)

    for k in range(kpos):
        hist, bin_edges = np.histogram(Vpica[:, k], nbins[k])
        x_dist: np.ndarray = np.linspace(bin_edges.min(), bin_edges.max(), num=100)
        area_hist: np.ndarray = Npos * (bin_edges[2] - bin_edges[1])
        scaled_pdf.append(area_hist * t.pdf(x_dist, *all_fits[k]))
        cd: np.ndarray = t.cdf(x_dist, *all_fits[k])
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
    np.fill_diagonal(Csca_nodiag, 0.)
    for k in range(kpos):
        icpos_tmp: list = ic_init[k].tolist()
        for kprime in (kp for kp in range(kpos) if (kp != k)):
            tmp: list = np.intersect1d(icpos_tmp, ic_init[kprime]).tolist()
            for i in tmp:
                remsec = (np.linalg.norm(Csca_nodiag[i, ic_init[k]])
                          < np.linalg.norm(Csca_nodiag[i, ic_init[kprime]]))
                if remsec:
                    icpos_tmp.remove(i)
        sortedpos += sorted(icpos_tmp, key=lambda i: -Vpica[i,k])
        icsize.append(len(icpos_tmp))
        s: Unit = Unit()
        s.items: np.ndarray = sorted(icpos_tmp, key=lambda i: -Vpica[i,k])
        s.col: float = k / kpos
        s.vect: np.ndarray = -Vpica[s.items, k]
        ics.append(s)
    return ics, icsize, sortedpos, cutoff, scaled_pdf, all_fits
