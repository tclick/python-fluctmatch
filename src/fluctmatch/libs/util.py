# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding: utf-8 -*-
# vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4
#
# fluctmatch --- https://github.com/tclick/python-fluctmatch
# Copyright (c) 2015-2017 The pySCA Development Team and contributors
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
import io
import logging
import typing
from collections.abc import Container, Iterable

import numpy as np
from sklearn.preprocessing import scale
from sklearn.utils import check_array

logger = logging.getLogger(__name__)


def iterable(obj: typing.Any) -> bool:
    """Determines if a value is iterable and not a string or stream.

    Parameters
    ----------
    obj
        Anything to test

    Returns
    -------
    True if the value is not an iterable container that is not a string or a
    stream
    """
    return (
        not isinstance(obj, str) and
        not isinstance(obj, io.IOBase) and
        isinstance(obj, Container) and
        isinstance(obj, Iterable)
    )


def asiterable(obj: typing.Any) -> typing.Tuple:
    """Create an iterable object.

    Parameters
    ----------
    obj
        Any object

    Returns
    -------
    Eiter the object, if already an iterable collection or a tuple of the object
    """
    return obj if iterable(obj) else (obj, )


def center2D(X: np.ndarray) -> np.ndarray:
    """Subtract the mean from both columns and rows.
    
    Parameters
    ----------
    X : array-like
        Any 2-D array
    
    Returns
    -------
    X_new : array-like
        A 2-D array with mean in both the columns and the rows.
    mean : array-like
         A 2-D array of the mean
    """
    X_new = check_array(
        X, copy=True, ensure_min_samples=2, ensure_min_features=2
    )
    X_new = scale(X_new, with_std=False, axis=0)
    X_new = scale(X_new, with_std=False, axis=1)
    mean = X - X_new
    return X_new, mean