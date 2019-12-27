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
import MDAnalysis as mda
import numpy as np
from numpy import testing

from fluctmatch.libs import fluctmatch as fmutils

from ..datafiles import TPR
from ..datafiles import XTC


def test_average_structure():
    universe = mda.Universe(TPR, XTC)
    avg_positions = np.mean(
        [universe.atoms.positions for _ in universe.trajectory], axis=0)
    positions = fmutils.AverageStructure(universe.atoms).run().result
    testing.assert_allclose(
        positions,
        avg_positions,
        err_msg="Average coordinates don't match.",
    )


def test_bond_stats():
    universe = mda.Universe(TPR, XTC)
    avg_bonds = np.mean(
        [universe.bonds.bonds() for _ in universe.trajectory], axis=0)
    bond_fluct = np.std(
        [universe.bonds.bonds() for _ in universe.trajectory], axis=0)
    bonds = fmutils.BondStats(universe.atoms).run().result
    testing.assert_allclose(
        bonds.average,
        avg_bonds,
        err_msg="Average bond distances don't match.",
    )
    testing.assert_allclose(
        bonds.stddev,
        bond_fluct,
        err_msg="Bond fluctuations don't match.",
    )
