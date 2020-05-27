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
from typing import List, Union

import MDAnalysis as mda
import numpy as np
import pandas as pd
from MDAnalysis.core.topologyobjects import TopologyGroup

_HEADER: List[str] = [
    "segidI",
    "resI",
    "I",
    "segidJ",
    "resJ",
    "J",
    "segidK",
    "resK",
    "K",
    "segidL",
    "resL",
    "L",
    "r_IJ",
    "T_IJK",
    "P_IJKL",
    "T_JKL",
    "r_KL",
]

logger: logging.Logger = logging.getLogger(__name__)


def create_empty_table(universe: Union[mda.Universe, mda.AtomGroup]) -> pd.DataFrame:
    """Create an empty table of internal coordinates from an atomgroup

    Parameters
    ----------
    universe : :class:`~MDAnalysis.Universe` or :class:`~MDAnalysis.AtomGroup`
        A collection of atoms in a universe or atomgroup with bond definitions.

    Returns
    -------
    A :class:`~pandas.DataFrame` compliant with a CHARMM-formatted internal
    coordinates (IC) table. The table matches the 'resid' version of an IC
    table.
    """
    table: pd.DataFrame = pd.DataFrame()
    atomgroup: mda.AtomGroup = universe.atoms
    logging.info("Creating an empty table.")
    try:
        dihedrals: TopologyGroup = atomgroup.dihedrals
        if len(dihedrals) == 0:
            raise AttributeError
    except AttributeError:
        try:
            angles: TopologyGroup = atomgroup.angles
            if len(angles) == 0:
                raise AttributeError
        except AttributeError:
            try:
                bonds: TopologyGroup = atomgroup.bonds
                if len(bonds) == 0:
                    raise AttributeError
            except AttributeError:
                tb = traceback.format_exc()
                msg = "Bonds, angles, and torsions undefined"
                logger.exception(msg)
                AttributeError(msg).with_traceback(tb)
            else:
                n_bonds: int = len(bonds)
                atom1, atom2 = bonds.atom1, bonds.atom2
                zeros: pd.DataFrame = pd.DataFrame(
                    np.zeros((n_bonds, 5), dtype=np.float)
                )
                cols: pd.DataFrame = pd.DataFrame(
                    [
                        atom1.segids,
                        atom1.resnums,
                        atom1.names,
                        atom2.segids,
                        atom2.resnums,
                        atom2.names,
                        ["??"] * n_bonds,
                        ["??"] * n_bonds,
                        ["??"] * n_bonds,
                        ["??"] * n_bonds,
                        ["??"] * n_bonds,
                        ["??"] * n_bonds,
                    ]
                ).T
                table: pd.DataFrame = pd.concat([table, cols, zeros], axis=1)
        else:
            n_angles: int = len(angles)
            atom1, atom2, atom3 = angles.atom1, angles.atom2, angles.atom3
            zeros: pd.DataFrame = pd.DataFrame(np.zeros((n_angles, 5), dtype=np.float))
            cols: pd.DataFrame = pd.DataFrame(
                [
                    atom1.segids,
                    atom1.resnums,
                    atom1.names,
                    atom2.segids,
                    atom2.resnums,
                    atom2.names,
                    atom3.segids,
                    atom3.resnums,
                    atom3.names,
                    ["??"] * n_angles,
                    ["??"] * n_angles,
                    ["??"] * n_angles,
                ]
            ).T
            table: pd.DataFrame = pd.concat([table, cols, zeros], axis=1)
    else:
        n_dihedrals: int = len(dihedrals)
        atom1, atom2, atom3, atom4 = (
            dihedrals.atom1,
            dihedrals.atom2,
            dihedrals.atom3,
            dihedrals.atom4,
        )
        zeros: pd.DataFrame = pd.DataFrame(np.zeros((n_dihedrals, 5), dtype=np.float))
        cols: pd.DataFrame = pd.DataFrame(
            [
                atom1.segids,
                atom1.resnums,
                atom1.names,
                atom2.segids,
                atom2.resnums,
                atom2.names,
                atom3.segids,
                atom3.resnums,
                atom3.names,
                atom4.segids,
                atom4.resnums,
                atom4.names,
            ]
        ).T
        table: pd.DataFrame = pd.concat([table, cols, zeros], axis=1)

    table.columns = _HEADER
    return table
