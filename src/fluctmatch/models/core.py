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
import logging
import traceback
from typing import List

import MDAnalysis as mda

from .. import _MODELS
from .base import Merge

logger = logging.getLogger(__name__)


def modeller(*args, **kwargs) -> mda.Universe:
    """Create coarse-grain model from universe selection.

    Parameters
    ----------
    args
    kwargs

    Returns
    -------
    A coarse-grain model
    """
    models: List[str] = [_.upper() for _ in kwargs.pop("model", ["polar",])]
    try:
        if "ENM" in models:
            logger.warning(
                "ENM model detected. All other models are being ignored."
            )
            universe = _MODELS["ENM"](*args, **kwargs)
            return universe
    except Exception as exc:
        logger.exception(
            "An error occurred while trying to create the universe."
        )
        raise RuntimeError from exc

    try:
        universe: List[mda.Universe] = [
            _MODELS[_](*args, **kwargs)
            for _ in models
        ]
    except KeyError:
        tb: List[str] = traceback.format_exc()
        msg = (
            f"One of the models is not implemented. Please try {_MODELS.keys()}"
        )
        logger.exception(msg)
        raise KeyError(msg).with_traceback(tb)
    else:
        universe: mda.Universe = Merge(*universe) if len(universe) > 1 else universe[0]
        return universe
