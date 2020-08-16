# fluctmatch --- https://github.com/tclick/python-fluctmatch
# Copyright (c) 2013-2017 The fluctmatch Development Team and contributors
# (see the file AUTHORS for the full list of names)
#
# Released under the New BSD license.
#
# Please cite your use of fluctmatch in published work:
#
# Timothy H. Click, Nixon Raj, and Jhih-Wei Chuniverse.
# Calculation of Enzyme Fluctuograms from All-Atom Molecular Dynamics
# Simulation. Meth Enzymology. 578 (2016), 327-342,
# doi:10.1016/bs.mie.2016.05.024.
"""Tests for the model base module."""

from typing import Tuple

import MDAnalysis as mda
import numpy as np
import pytest
from numpy import testing

from fluctmatch.core.base import Merge, ModelBase, rename_universe
from tests.datafiles import TPR, XTC


def test_universe():
    testing.assert_raises(TypeError, ModelBase)


class TestMerge:
    @pytest.fixture(scope="class")
    def universe(self) -> mda.Universe:
        return mda.Universe(TPR, XTC)

    def test_creation(self, universe: mda.Universe) -> None:
        univ_tuple: Tuple[mda.Universe, ...] = (universe, universe)
        new_universe: mda.Universe = Merge(*univ_tuple)

        n_atoms = universe.atoms.n_atoms * 2
        assert new_universe.atoms.n_atoms == n_atoms, "Number of sites don't match."

    def test_positions(self, universe: mda.Universe) -> None:
        n_atoms = universe.atoms.n_atoms
        univ_tuple: Tuple[mda.Universe, ...] = (universe, universe)
        new_universe: mda.Universe = Merge(*univ_tuple)

        positions = np.concatenate([universe.atoms.positions for u in univ_tuple])

        testing.assert_allclose(
            new_universe.atoms.positions, positions, err_msg="Coordinates don't match."
        )
        testing.assert_allclose(
            new_universe.atoms.positions[0],
            new_universe.atoms.positions[n_atoms],
            err_msg=f"Coordinates 0 and {n_atoms:d} " f"don't match.",
        )

    def test_topology(self, universe: mda.Universe) -> None:
        new_universe: mda.Universe = Merge(universe)

        assert universe.atoms.n_atoms == new_universe.atoms.n_atoms
        assert new_universe.bonds == universe.bonds, "Bonds differ."
        assert new_universe.angles == universe.angles, "Angles differ."
        assert new_universe.dihedrals == universe.dihedrals, "Dihedrals differ."


def test_rename_universe() -> None:
    universe: mda.Universe = mda.Universe(TPR, XTC)
    rename_universe(universe)

    testing.assert_string_equal(universe.atoms[0].name, "A001")
    testing.assert_string_equal(universe.atoms[-1].name, "F001")
    testing.assert_string_equal(universe.residues[0].resname, "A001")
