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

from typing import Tuple

import MDAnalysis as mda
from numpy import testing
from fluctmatch.models.base import ModelBase, Merge, rename_universe
from fluctmatch.models.selection import *
from tests.datafiles import TPR, XTC


def test_universe():
    testing.assert_raises(TypeError, ModelBase)


def test_merge_creation():
    universe: mda.Universe = mda.Universe(TPR, XTC)
    univ_tuple: Tuple[mda.Universe, mda.Universe] = (universe, universe)
    new_universe = Merge(*univ_tuple)

    n_atoms: int = universe.atoms.n_atoms * 2
    testing.assert_equal(new_universe.atoms.n_atoms, n_atoms,
                         err_msg="Number of sites don't match.", verbose=True)


def test_merge_positions():
    universe: mda.Universe = mda.Universe(TPR, XTC)
    n_atoms: int = universe.atoms.n_atoms
    univ_tuple: Tuple[mda.Universe, mda.Universe] = (universe, universe)
    new_universe = Merge(*univ_tuple)

    positions = np.concatenate([u.atoms.positions for u in univ_tuple], axis=0)
    testing.assert_allclose(new_universe.atoms.positions, positions,
        err_msg="Coordinates don't match.")
    testing.assert_allclose(new_universe.atoms.positions[0],
                            new_universe.atoms.positions[n_atoms],
                            err_msg=f"Coordinates 0 and {n_atoms:d} don't match.")


def test_merge_topology():
    universe: mda.Universe = mda.Universe(TPR, XTC)
    new_universe = Merge(universe)

    assert new_universe.bonds == universe.bonds
    assert new_universe.angles == universe.angles
    assert new_universe.dihedrals == universe.dihedrals


def test_rename_universe():
    universe: mda.Universe = mda.Universe(TPR, XTC)
    n_atoms: int = universe.atoms.n_atoms
    rename_universe(universe)
    testing.assert_string_equal(universe.atoms[0].name, "A001")
    testing.assert_string_equal(universe.atoms[-1].name, "F001")
    testing.assert_string_equal(universe.residues[0].resname, "A001")


def test_registry():
    from fluctmatch import _MODELS
    assert len(_MODELS) > 0
