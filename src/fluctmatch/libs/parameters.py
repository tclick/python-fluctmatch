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
"""Utilities for parameter creation."""

from typing import Dict
from typing import List
from typing import Mapping
from typing import Union

import MDAnalysis as mda
import numpy as np
import pandas as pd


def create_empty_parameters(
    universe: Union[mda.Universe, mda.AtomGroup], **kwargs: Mapping
) -> Dict[str, pd.DataFrame]:
    """

    Parameters
    ----------
    universe : :class:`~MDAnalysis.Universe` or :class:`~MDAnalysis.AtomGroup`
        A collection of atoms in a universe or atomgroup with bond definitions.
    charmm
        Allow for automatic atom typing (type set to -1).

    Returns
    -------
    An dictionary with keys defining the CHARMM parameter sections, and each
    associated value contianing a :class:`~pandas.DataFrame` with a table
    containing the necessary information. Tables for bonds, angles, dihedrals,
    and impropers will have values of 0 that can be filled by the user.
    """
    version: int = kwargs.get("charmm_version", 41)
    parameters: Dict[str, pd.DataFrame] = dict(
        ATOMS=pd.DataFrame(),
        BONDS=pd.DataFrame(),
        ANGLES=pd.DataFrame(),
        DIHEDRALS=pd.DataFrame(),
        IMPROPER=pd.DataFrame(),
    )
    param_columns: Dict[str, List[str]] = dict(
        ATOMS=["type", "atom", "mass"],
        BONDS=["I", "J", "Kb", "b0"],
        ANGLES=["I", "J", "K", "Ktheta", "theta0"],
        DIHEDRALS=["I", "J", "K", "L", "Kchi", "n", "delta"],
    )

    # Atoms
    types: np.ndarray = (
        universe.atoms.types[:, None]
        if np.issubdtype(universe.atoms.types.dtype, np.int)
        else np.arange(universe.atoms.n_atoms) + 1
    )[:, None]
    atoms: np.ndarray = np.hstack(
        (types, universe.atoms.names[:, None], universe.atoms.masses[:, None])
    )
    parameters["ATOMS"] = pd.DataFrame(atoms, columns=param_columns["ATOMS"])
    if version > 39:
        parameters["ATOMS"]["type"]: int = -1

    # Bonds
    try:
        bonds = np.hstack(
            (
                universe.bonds.atom1.names[:, None],
                universe.bonds.atom2.names[:, None],
                np.zeros((universe.bonds.atom1.names.size, 2), dtype=np.float),
            )
        )
        parameters["BONDS"] = pd.DataFrame(bonds, columns=param_columns["BONDS"])
    except (mda.NoDataError, AttributeError, IndexError):
        pass

    # Angles
    try:
        angles = np.hstack(
            (
                universe.angles.atom1.names[:, None],
                universe.angles.atom2.names[:, None],
                universe.angles.atom3.names[:, None],
                np.zeros((universe.angles.atom1.names.size, 2), dtype=np.float),
            )
        )
        parameters["ANGLES"] = pd.DataFrame(
            angles, columns=param_columns["ANGLES"]
        )
    except (mda.NoDataError, AttributeError, IndexError):
        pass

    # Dihedrals
    try:
        dihedrals = np.hstack(
            (
                universe.dihedrals.atom1.names[:, None],
                universe.dihedrals.atom2.names[:, None],
                universe.dihedrals.atom3.names[:, None],
                universe.dihedrals.atom4.names[:, None],
                np.zeros(
                    (universe.dihedrals.atom1.names.size, 1), dtype=np.float
                ),
                np.zeros((universe.dihedrals.atom1.names.size, 1), dtype=np.int),
                np.zeros(
                    (universe.dihedrals.atom1.names.size, 1), dtype=np.float
                ),
            )
        )
        parameters["DIHEDRALS"] = pd.DataFrame(
            dihedrals, columns=param_columns["DIHEDRALS"]
        )
    except (mda.NoDataError, AttributeError, IndexError):
        pass

    # Impropers
    try:
        impropers = np.hstack(
            (
                universe.impropers.atom1.names[:, None],
                universe.impropers.atom2.names[:, None],
                universe.impropers.atom3.names[:, None],
                universe.impropers.atom4.names[:, None],
                np.zeros(
                    (universe.impropers.atom1.names.size, 1), dtype=np.float
                ),
                np.zeros((universe.impropers.atom1.names.size, 1), dtype=np.int),
                np.zeros(
                    (universe.impropers.atom1.names.size, 1), dtype=np.float
                ),
            )
        )
        parameters["IMPROPER"] = pd.DataFrame(
            impropers, columns=param_columns["DIHEDRALS"]
        )
    except (mda.NoDataError, AttributeError, IndexError):
        pass

    return parameters
