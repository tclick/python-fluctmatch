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

import numpy as np
import MDAnalysis as mda
import pandas as pd
from MDAnalysis.core.topologyobjects import TopologyGroup

from ..libs.typing import MDUniverse

_HEADER: List[str] = [
    "segidI", "resI", "I", "segidJ", "resJ", "J", "segidK", "resK", "K",
    "segidL", "resL", "L", "r_IJ", "T_IJK", "P_IJKL", "T_JKL", "r_KL"
]

logger: logging.Logger = logging.getLogger(__name__)


def create_empty_table(universe: MDUniverse) -> pd.DataFrame:
    """Create an empty table of internal coordinates from an atomgroup

    Parameters
    ----------
    universe : :class:`~MDAnalysis.Universe` or :class:`~MDAnalysis.AtomGroup`
        A collection of atoms in a universe or atomgroup with bond definitions.

    Returns
    -------
    A :class:`~pandas.DataFrame` compliant with a CHARMM-formatted internal
    coordinates (IC) table. The table matches the 'resid' version of an IC table.
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
                zeros: pd.DataFrame = pd.DataFrame(np.zeros((n_bonds, 5), dtype=np.float))
                cols: pd.DataFrame = pd.DataFrame([
                    atom1.segids, atom1.resnums, atom1.names, atom2.segids,
                    atom2.resnums, atom2.names, ["??",] * n_bonds,
                    ["??",] * n_bonds, ["??",] * n_bonds, ["??",] * n_bonds,
                    ["??",] * n_bonds, ["??",] * n_bonds
                ]).T
                table: pd.DataFrame = pd.concat([table, cols, zeros], axis=1)
        else:
            n_angles: int = len(angles)
            atom1, atom2, atom3 = angles.atom1, angles.atom2, angles.atom3
            zeros: pd.DataFrame = pd.DataFrame(np.zeros((n_angles, 5), dtype=np.float))
            cols: pd.DataFrame = pd.DataFrame([
                atom1.segids, atom1.resnums, atom1.names, atom2.segids,
                atom2.resnums, atom2.names, atom3.segids, atom3.resnums,
                atom3.names, ["??",] * n_angles, ["??",] * n_angles,
                ["??",] * n_angles
            ]).T
            table: pd.DataFrame = pd.concat([table, cols, zeros], axis=1)
    else:
        n_dihedrals: int = len(dihedrals)
        atom1, atom2, atom3, atom4 = (
            dihedrals.atom1, dihedrals.atom2,dihedrals.atom3, dihedrals.atom4
        )
        zeros: pd.DataFrame = pd.DataFrame(np.zeros((n_dihedrals, 5), dtype=np.float))
        cols: pd.DataFrame = pd.DataFrame([
            atom1.segids, atom1.resnums, atom1.names, atom2.segids,
            atom2.resnums, atom2.names, atom3.segids, atom3.resnums,
            atom3.names, atom4.segids, atom4.resnums, atom4.names
        ]).T
        table: pd.DataFrame = pd.concat([table, cols, zeros], axis=1)

    table.columns: pd.Index = _HEADER
    return table
