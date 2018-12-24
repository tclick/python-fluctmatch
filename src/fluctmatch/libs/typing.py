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
import logging
from pathlib import Path
from typing import TypeVar

import MDAnalysis as mda
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Define a type checker
Array = TypeVar("Array", np.ndarray, np.matrix, pd.DataFrame)
FileName = TypeVar("FileName", str, Path)
MDUniverse = TypeVar("MDUniverse", mda.Universe, mda.AtomGroup)
