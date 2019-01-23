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
"""Tests for the model base module."""

from typing import Tuple

import MDAnalysis as mda
from numpy import testing
import pytest

from fluctmatch.models.base import ModelBase, Merge, rename_universe
from fluctmatch.models.selection import *
from tests.datafiles import TPR, XTC


def test_universe():
    testing.assert_raises(TypeError, ModelBase)


class TestMerge:
    @pytest.fixture()
    def u(self) -> mda.Universe:
        return mda.Universe(TPR, XTC)

    def test_creation(self, u: mda.Universe):
        univ_tuple: Tuple[mda.Universe, mda.Universe] = (u, u)
        u2: mda.Universe = Merge(*univ_tuple)
    
        n_atoms: int = u.atoms.n_atoms * 2
        testing.assert_equal(u2.atoms.n_atoms, n_atoms,
                             err_msg="Number of sites don't match.")
    
    def test_positions(self, u: mda.Universe):
        n_atoms: int = u.atoms.n_atoms
        univ_tuple: Tuple[mda.Universe, mda.Universe] = (u, u)
        u2: mda.Universe = Merge(*univ_tuple)
    
        positions: np.ndarray = np.concatenate(
            [u.atoms.positions 
             for u in univ_tuple])
        
        testing.assert_allclose(u2.atoms.positions, positions,
                                err_msg="Coordinates don't match.")
        testing.assert_allclose(u2.atoms.positions[0],
                                u2.atoms.positions[n_atoms],
                                err_msg=(f"Coordinates 0 and {n_atoms:d} "
                                         f"don't match."))
    
    def test_topology(self, u: mda.Universe):
        u2: mda.Universe = Merge(u)
    
        testing.assert_equal(u.atoms.n_atoms, u2.atoms.n_atoms)
        testing.assert_equal(u2.bonds, u.bonds, err_msg="Bonds differ.")
        testing.assert_equal(u2.angles, u.angles, err_msg="Angles differ.")
        testing.assert_equal(u2.dihedrals, u.dihedrals,
                             err_msg="Dihedrals differ.")


def test_rename_universe():
    universe: mda.Universe = mda.Universe(TPR, XTC)
    rename_universe(universe)

    testing.assert_string_equal(universe.atoms[0].name, "A001")
    testing.assert_string_equal(universe.atoms[-1].name, "F001")
    testing.assert_string_equal(universe.residues[0].resname, "A001")


def test_registry():
    from fluctmatch import _MODELS
    assert len(_MODELS) > 0
