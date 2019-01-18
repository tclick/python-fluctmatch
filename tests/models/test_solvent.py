# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding: utf-8 -*-
#
#  python-fluctmatch -
#  Copyright (c) 2019 Timothy H. Click, Ph.D.
#
#  All rights reserved.
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:
#
#  Redistributions of source code must retain the above copyright notice, this
#  list of conditions and the following disclaimer.
#
#  Redistributions in binary form must reproduce the above copyright notice,
#  this list of conditions and the following disclaimer in the documentation
#  and/or other materials provided with the distribution.
#
#  Neither the name of the author nor the names of its contributors may be used
#  to endorse or promote products derived from this software without specific
#  prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS “AS IS”
#  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
#  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
#  ARE DISCLAIMED. IN NO EVENT SHALL THE REGENTS OR CONTRIBUTORS BE LIABLE FOR
#  ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
#  DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
#  SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
#  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
#  OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
#  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
"""Tests for different solvent models."""

import MDAnalysis as mda
from numpy import testing

from fluctmatch.models import solvent
from fluctmatch.models.selection import *
from ..datafiles import TIP3P, TIP4P, DMA


def test_water_from_tip3p_creation():
    aa_universe: mda.Universe = mda.Universe(TIP3P)
    water: solvent.Water = solvent.Water()
    water.create_topology(aa_universe)

    n_atoms: int = sum(aa_universe.select_atoms(selection).residues.n_residues
                       for selection in water._mapping.values())

    testing.assert_equal(water.universe.atoms.n_atoms, n_atoms,
                         err_msg="Number of sites don't match.", verbose=True)


def test_water_from_tip3p_positions():
    aa_universe: mda.Universe = mda.Universe(TIP3P)
    water: solvent.Water = solvent.Water()
    cg_universe: mda.Universe = water.transform(aa_universe)

    positions: np.ndarray = np.asarray([
        _.atoms.select_atoms(selection).center_of_mass()
        for _ in aa_universe.select_atoms("water").residues
        for selection in water._mapping.values()
        if _.atoms.select_atoms(selection)
    ])

    testing.assert_allclose(
        np.asarray(positions),
        cg_universe.atoms.positions,
        err_msg="The coordinates do not match.",
    )


def test_water_from_tip3p_bonds():
    aa_universe: mda.Universe = mda.Universe(TIP3P)
    water: solvent.Water = solvent.Water()
    cg_universe: mda.Universe = water.transform(aa_universe)

    testing.assert_equal(len(cg_universe.bonds), 0,
                         err_msg="No bonds should exist.")


def test_water_mass():
    aa_universe: mda.Universe = mda.Universe(TIP3P)
    water: solvent.Water = solvent.Water()
    cg_universe: mda.Universe = water.transform(aa_universe)

    masses: np.ndarray = np.fromiter([
        _.atoms.select_atoms(selection).total_mass()
        for _ in aa_universe.select_atoms("water").residues
        for selection in water._selection.values()
        if _.atoms.select_atoms(selection)
    ], dtype=np.float32)

    testing.assert_allclose(cg_universe.atoms.masses, masses,
                            err_msg="The masses do not match.")


def test_water_charges():
    aa_universe: mda.Universe = mda.Universe(TIP3P)
    water: solvent.Water = solvent.Water()
    cg_universe: mda.Universe = water.transform(aa_universe)

    testing.assert_allclose(cg_universe.atoms.charges,
                            np.zeros(cg_universe.atoms.n_atoms),
                            err_msg="The masses do not match.")


def test_water_from_tip4p_creation():
    aa_universe: mda.Universe = mda.Universe(TIP4P)
    water: solvent.Water = solvent.Water()
    water.create_topology(aa_universe)

    n_atoms: int = sum(aa_universe.select_atoms(selection).residues.n_residues
                       for selection in water._mapping.values())

    testing.assert_equal(water.universe.atoms.n_atoms, n_atoms,
                         err_msg="Number of sites don't match.", verbose=True)


def test_water_from_tip4p_positions():
    aa_universe: mda.Universe = mda.Universe(TIP4P)
    water: solvent.Water = solvent.Water()
    cg_universe: mda.Universe = water.transform(aa_universe)

    positions: np.ndarray = np.asarray([
        _.atoms.select_atoms(selection).center_of_mass()
        for _ in aa_universe.select_atoms("water").residues
        for selection in water._mapping.values()
        if _.atoms.select_atoms(selection)
    ])

    testing.assert_allclose(
        np.asarray(positions),
        cg_universe.atoms.positions,
        err_msg="The coordinates do not match.",
    )


def test_water_from_tip4p_bonds():
    aa_universe: mda.Universe = mda.Universe(TIP4P)
    water: solvent.Water = solvent.Water()
    cg_universe: mda.Universe = water.transform(aa_universe)

    testing.assert_equal(len(cg_universe.bonds), 0,
                         err_msg="No bonds should exist.")


def test_tip3p_creation():
    aa_universe: mda.Universe = mda.Universe(TIP3P)
    water: solvent.Water = solvent.Tip3p()
    water.create_topology(aa_universe)

    n_atoms: int = sum(aa_universe.select_atoms(selection).residues.n_residues
                       for selection in water._mapping.values())

    testing.assert_equal(water.universe.atoms.n_atoms, n_atoms,
                         err_msg="Number of sites don't match.", verbose=True)


def test_tip3p_positions():
    aa_universe: mda.Universe = mda.Universe(TIP4P)
    water: solvent.Water = solvent.Tip3p()
    cg_universe: mda.Universe = water.transform(aa_universe)

    positions: np.ndarray = np.asarray([
        _.atoms.select_atoms(selection).center_of_mass()
        for _ in aa_universe.select_atoms("water").residues
        for selection in water._mapping.values()
        if _.atoms.select_atoms(selection)
    ])

    testing.assert_allclose(
        np.asarray(positions),
        cg_universe.atoms.positions,
        err_msg="The coordinates do not match.",
    )


def test_tip3p_bonds():
    aa_universe: mda.Universe = mda.Universe(TIP4P)
    water: solvent.Water = solvent.Tip3p()
    cg_universe: mda.Universe = water.transform(aa_universe)

    testing.assert_equal(len(cg_universe.bonds),
                         water.universe.residues.n_residues * 3,
                         err_msg="Expected and actual number of bonds not equal")
    testing.assert_equal(len(cg_universe.angles),
                         water.universe.residues.n_residues * 3,
                         err_msg="Expected and actual number of angles not equal")
    testing.assert_equal(len(cg_universe.dihedrals), 0,
                         err_msg="No dihedral angles should exist.")
    testing.assert_equal(len(cg_universe.impropers), 0,
                         err_msg="No improper dihedral angles should exist.")


def test_dma_creation():
    aa_universe: mda.Universe = mda.Universe(DMA)
    dma: solvent.Dma = solvent.Dma()
    dma.create_topology(aa_universe)

    n_atoms: int = sum(aa_universe.select_atoms(selection).residues.n_residues
                       for selection in dma._mapping.values())

    testing.assert_equal(dma.universe.atoms.n_atoms, n_atoms,
                         err_msg="Number of sites don't match.", verbose=True)


def test_dma_positions():
    aa_universe: mda.Universe = mda.Universe(DMA)
    dma: solvent.Dma = solvent.Dma()
    cg_universe: mda.Universe = dma.transform(aa_universe)

    positions: np.ndarray = np.asarray([
        _.atoms.select_atoms(selection).center_of_mass()
        for _ in aa_universe.select_atoms("resname DMA").residues
        for selection in dma._mapping.values()
        if _.atoms.select_atoms(selection)
    ])

    testing.assert_allclose(
        positions, cg_universe.atoms.positions,
        err_msg="The coordinates do not match.",
    )


def test_dma_bonds():
    aa_universe: mda.Universe = mda.Universe(DMA)
    dma: solvent.Water = solvent.Dma()
    cg_universe: mda.Universe = dma.transform(aa_universe)

    testing.assert_equal(len(cg_universe.bonds),
                         dma.universe.residues.n_residues * 3,
                         err_msg="Expected and actual number of bonds not equal")
    testing.assert_equal(len(cg_universe.angles),
                         dma.universe.residues.n_residues * 3,
                         err_msg="Expected and actual number of angles not equal")
    testing.assert_equal(len(cg_universe.dihedrals), 0,
                         err_msg="No dihedral angles should exist.")
    testing.assert_equal(len(cg_universe.impropers),
                         dma.universe.residues.n_residues * 3,
                         err_msg=("Expected and actual number of improper "
                                  "dihedral angles not equal."))
