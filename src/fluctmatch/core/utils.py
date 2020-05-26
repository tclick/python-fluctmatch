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

import logging
import traceback
from typing import List

import MDAnalysis as mda

from .. import _MODELS
from .base import Merge
from .base import ModelBase

logger: logging.Logger = logging.getLogger(__name__)


def modeller(*args, **kwargs) -> mda.Universe:
    """Create coarse-grain model from universe selection.

    Parameters
    ----------
    topology : str
        A topology file containing atomic information about a system.
    trajectory : str
        A trajectory file with coordinates of atoms
    model : list[str], optional
        Name(s) of coarse-grain core

    Returns
    -------
    A coarse-grain model
    """
    models: List[str] = [_.upper() for _ in kwargs.pop("model", ["polar"])]
    try:
        if "ENM" in models:
            logger.warning(
                "ENM model detected. All other core are " "being ignored."
            )
            model: ModelBase = _MODELS["ENM"](**kwargs)
            return model.transform(mda.Universe(*args, **kwargs))
    except Exception as exc:
        logger.exception("An error occurred while trying to create the universe.")
        raise RuntimeError from exc

    try:
        universe: List[mda.Universe] = [
            _MODELS[_]().transform(mda.Universe(*args)) for _ in models
        ]
    except KeyError:
        tb: List[str] = traceback.format_exc()
        msg: str = f"One of the core is not implemented. Please try {_MODELS.keys()}"
        logger.exception(msg)
        raise KeyError(msg).with_traceback(tb)
    else:
        return Merge(*universe)
