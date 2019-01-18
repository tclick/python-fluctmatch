# -*- coding: utf-8 -*-
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
#  Timothy H. Click, Nixon Raj, and Jhih-Wei Chu.
#  Simulation. Meth Enzymology. 578 (2016), 327-342,
#  Calculation of Enzyme Fluctuograms from All-Atom Molecular Dynamics
#  doi:10.1016/bs.mie.2016.05.024.
"""Tests for various protein models."""

from typing import List

import MDAnalysis as mda
from numpy import testing

from fluctmatch.models import protein
from fluctmatch.models.selection import *
from ..datafiles import TPR, XTC


def test_calpha_creation():
    aa_universe: mda.Universe = mda.Universe(TPR, XTC)
    system: protein.Calpha = protein.Calpha()
    system.create_topology(aa_universe)

    n_atoms = aa_universe.select_atoms("calpha or bioion").n_atoms
    testing.assert_equal(system.universe.atoms.n_atoms, n_atoms,
                         err_msg="Number of sites don't match.")


def test_calpha_positions():
    aa_universe: mda.Universe = mda.Universe(TPR, XTC)
    system: protein.Calpha = protein.Calpha()
    cg_universe: mda.Universe = system.transform(aa_universe)

    positions: np.ndarray = np.asarray([
        residue.atoms.select_atoms(sel).center_of_mass()
        for residue in aa_universe.select_atoms("protein or bioion").residues
        for sel in system._mapping.values()
        if residue.atoms.select_atoms(sel)
    ])

    testing.assert_allclose(
        cg_universe.atoms.positions,
        positions,
        err_msg="The coordinates do not match.")


def test_calpha_trajectory():
    aa_universe: mda.Universe = mda.Universe(TPR, XTC)
    system: protein.Calpha = protein.Calpha()
    cg_universe: mda.Universe = system.transform(aa_universe)

    testing.assert_equal(
        cg_universe.trajectory.n_frames,
        aa_universe.trajectory.n_frames,
        err_msg="All-atom and coarse-grain trajectories unequal.")


def test_caside_creation():
    aa_universe: mda.Universe = mda.Universe(TPR, XTC)
    system: protein.Caside = protein.Caside()
    system.create_topology(aa_universe)

    n_atoms = aa_universe.select_atoms("calpha or cbeta or bioion").n_atoms
    testing.assert_equal(system.universe.atoms.n_atoms, n_atoms,
                         err_msg="Number of sites not equal.")


def test_caside_positions():
    aa_universe: mda.Universe = mda.Universe(TPR, XTC)
    system: protein.Caside = protein.Caside()
    cg_universe: mda.Universe = system.transform(aa_universe)

    positions: np.ndarray = np.asarray([
        residue.atoms.select_atoms(sel).center_of_mass()
        for residue in aa_universe.select_atoms("protein or bioion").residues
        for sel in system._mapping.values()
        if residue.atoms.select_atoms(sel)
    ])

    testing.assert_allclose(
        cg_universe.atoms.positions,
        positions,
        err_msg="The coordinates do not match.")


def test_caside_trajectory():
    aa_universe: mda.Universe = mda.Universe(TPR, XTC)
    system: protein.Calpha = protein.Calpha()
    cg_universe: mda.Universe = system.transform(aa_universe)

    testing.assert_equal(
        cg_universe.trajectory.n_frames,
        aa_universe.trajectory.n_frames,
        err_msg="All-atom and coarse-grain trajectories unequal.")


def test_ncsc_creation():
    aa_universe: mda.Universe = mda.Universe(TPR, XTC)
    system: protein.Ncsc = protein.Ncsc()
    system.create_topology(aa_universe)
    n_atoms = aa_universe.select_atoms(
        "(protein and name N O OT1) or cbeta or bioion").n_atoms
    testing.assert_equal(system.universe.atoms.n_atoms, n_atoms,
                         err_msg="Number of sites not equal.")


def test_ncsc_positions():
    aa_universe: mda.Universe = mda.Universe(TPR, XTC)
    system: protein.Ncsc = protein.Ncsc()
    cg_universe: mda.Universe = system.transform(aa_universe)

    positions: np.ndarray = np.asarray([
        residue.atoms.select_atoms(sel).center_of_mass()
        for residue in aa_universe.select_atoms("protein or bioion").residues
        for sel in system._mapping.values()
        if residue.atoms.select_atoms(sel)
    ])

    testing.assert_allclose(
        cg_universe.atoms.positions,
        positions,
        err_msg="The coordinates do not match.")


def test_ncsc_trajectory():
    aa_universe: mda.Universe = mda.Universe(TPR, XTC)
    system: protein.Ncsc = protein.Ncsc()
    cg_universe: mda.Universe = system.transform(aa_universe)

    testing.assert_equal(
        cg_universe.trajectory.n_frames,
        aa_universe.trajectory.n_frames,
        err_msg="All-atom and coarse-grain trajectories unequal.")


def test_polar_creation():
    aa_universe: mda.Universe = mda.Universe(TPR, XTC)
    system: protein.Polar = protein.Polar()
    system.create_topology(aa_universe)
    n_atoms = aa_universe.select_atoms(
        "(protein and name N O OT1) or cbeta or bioion").n_atoms
    testing.assert_equal(system.universe.atoms.n_atoms, n_atoms,
                         err_msg="Number of sites not equal.")


def test_polar_positions():
    aa_universe: mda.Universe = mda.Universe(TPR, XTC)
    system: protein.Ncsc = protein.Ncsc()
    cg_universe: mda.Universe = system.transform(aa_universe)
    beads: List[mda.AtomGroup] = []

    for residue in aa_universe.select_atoms("protein or bioion").residues:
        for sel in system._mapping.values():
            if isinstance(sel, dict):
                value: mda.AtomGroup = sel.get(
                    residue.resname, "hsidechain and not name H*")
                bead: mda.AtomGroup = residue.atoms.select_atoms(value)
            else:
                bead: mda.AtomGroup = residue.atoms.select_atoms(sel)
            if bead:
                beads.append(bead)

    positions: List[np.ndarray] = np.asarray([
        bead.center_of_mass()
        for bead in beads
        if bead
    ])

    testing.assert_allclose(cg_universe.atoms.positions, positions,
                            err_msg="The coordinates do not match.")


def test_polar_trajectory():
    aa_universe: mda.Universe = mda.Universe(TPR, XTC)
    system: protein.Polar = protein.Polar()
    cg_universe: mda.Universe = system.transform(aa_universe)

    testing.assert_equal(
        cg_universe.trajectory.n_frames,
        aa_universe.trajectory.n_frames,
        err_msg="All-atom and coarse-grain trajectories unequal.")


def test_ncsc_polar_positions():
    aa_universe: mda.Universe = mda.Universe(TPR, XTC)
    polar: protein.Polar = protein.Polar()
    polar_universe: mda.Universe = polar.transform(aa_universe)

    ncsc: protein.Ncsc = protein.Ncsc()
    ncsc_universe: mda.Universe = ncsc.transform(aa_universe)

    assert not np.allclose(polar_universe.atoms.positions,
                           ncsc_universe.atoms.positions)
