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

import io
import logging
import typing
from collections.abc import Container, Iterable

import numpy as np
from sklearn.preprocessing import scale
from sklearn.utils import check_array

logger: logging.Logger = logging.getLogger(__name__)


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
        not isinstance(obj, str)
        and not isinstance(obj, io.IOBase)
        and isinstance(obj, Container)
        and isinstance(obj, Iterable)
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
    return obj if iterable(obj) else (obj,)


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
    X_new = check_array(X, copy=True, ensure_min_samples=2, ensure_min_features=2)
    X_new = scale(X_new, with_std=False, axis=0)
    X_new = scale(X_new, with_std=False, axis=1)
    mean = X - X_new
    return X_new, mean
