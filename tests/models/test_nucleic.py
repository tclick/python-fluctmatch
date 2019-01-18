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
"""Tests for the different nucleic acid models."""

import MDAnalysis as mda
from numpy import testing

from fluctmatch.models import nucleic
from fluctmatch.models.selection import *
from ..datafiles import TPR, XTC


def test_nucleic3_creation():
    aa_universe: mda.Universe = mda.Universe(TPR, XTC)
    system: nucleic.Nucleic3 = nucleic.Nucleic3()
    system.create_topology(aa_universe)

    n_atoms = sum(aa_universe.select_atoms(sel).residues.n_residues
                  for sel in system._mapping.values())
    testing.assert_equal(system.universe.atoms.n_atoms, n_atoms,
                         err_msg="Number of sites don't match.")


def test_nucleic3_positions():
    aa_universe: mda.Universe = mda.Universe(TPR, XTC)
    system: nucleic.Nucleic3 = nucleic.Nucleic3()
    cg_universe: mda.Universe = system.transform(aa_universe)

    positions: np.ndarray = np.asarray([
        residue.atoms.select_atoms(sel).center_of_mass()
        for residue in aa_universe.select_atoms("nucleic or bioion").residues
        for sel in system._mapping.values()
        if residue.atoms.select_atoms(sel)
    ])

    testing.assert_allclose(
        cg_universe.atoms.positions,
        positions,
        err_msg="The coordinates do not match.")


def test_nucleic3_trajectory():
    aa_universe: mda.Universe = mda.Universe(TPR, XTC)
    system: nucleic.Nucleic3 = nucleic.Nucleic3()
    cg_universe: mda.Universe = system.transform(aa_universe)

    testing.assert_equal(
        cg_universe.trajectory.n_frames,
        aa_universe.trajectory.n_frames,
        err_msg="All-atom and coarse-grain trajectories unequal.")


def test_nucleic4_creation():
    aa_universe: mda.Universe = mda.Universe(TPR, XTC)
    system: nucleic.Nucleic4 = nucleic.Nucleic4()
    system.create_topology(aa_universe)

    n_atoms = sum(aa_universe.select_atoms(sel).residues.n_residues
                  for sel in system._mapping.values())
    testing.assert_equal(system.universe.atoms.n_atoms, n_atoms,
                         err_msg="Number of sites don't match.")


def test_nucleic4_positions():
    aa_universe: mda.Universe = mda.Universe(TPR, XTC)
    system: nucleic.Nucleic4 = nucleic.Nucleic4()
    cg_universe: mda.Universe = system.transform(aa_universe)

    positions: np.ndarray = np.asarray([
        residue.atoms.select_atoms(sel).center_of_mass()
        for residue in aa_universe.select_atoms("nucleic or bioion").residues
        for sel in system._mapping.values()
        if residue.atoms.select_atoms(sel)
    ])

    testing.assert_allclose(
        cg_universe.atoms.positions,
        positions,
        err_msg="The coordinates do not match.")


def test_nucleic4_trajectory():
    aa_universe: mda.Universe = mda.Universe(TPR, XTC)
    system: nucleic.Nucleic4 = nucleic.Nucleic4()
    cg_universe: mda.Universe = system.transform(aa_universe)

    testing.assert_equal(
        cg_universe.trajectory.n_frames,
        aa_universe.trajectory.n_frames,
        err_msg="All-atom and coarse-grain trajectories unequal.")


def test_nucleic6_creation():
    aa_universe: mda.Universe = mda.Universe(TPR, XTC)
    system: nucleic.Nucleic6 = nucleic.Nucleic6()
    system.create_topology(aa_universe)

    n_atoms = sum(aa_universe.select_atoms(sel).residues.n_residues
                  for sel in system._mapping.values())
    testing.assert_equal(system.universe.atoms.n_atoms, n_atoms,
                         err_msg="Number of sites don't match.")


def test_nucleic6_positions():
    aa_universe: mda.Universe = mda.Universe(TPR, XTC)
    system: nucleic.Nucleic6 = nucleic.Nucleic6()
    cg_universe: mda.Universe = system.transform(aa_universe)

    positions: np.ndarray = np.asarray([
        residue.atoms.select_atoms(sel).center_of_mass()
        for residue in aa_universe.select_atoms("nucleic or bioion").residues
        for sel in system._mapping.values()
        if residue.atoms.select_atoms(sel)
    ])

    testing.assert_allclose(
        cg_universe.atoms.positions,
        positions,
        err_msg="The coordinates do not match.")


def test_nucleic6_trajectory():
    aa_universe: mda.Universe = mda.Universe(TPR, XTC)
    system: nucleic.Nucleic6 = nucleic.Nucleic6()
    cg_universe: mda.Universe = system.transform(aa_universe)

    testing.assert_equal(
        cg_universe.trajectory.n_frames,
        aa_universe.trajectory.n_frames,
        err_msg="All-atom and coarse-grain trajectories unequal.")


def test_nucleic6_charges():
    aa_universe: mda.Universe = mda.Universe(TPR, XTC)
    system: nucleic.Nucleic6 = nucleic.Nucleic6()
    system.create_topology(aa_universe)

    charges: np.ndarray = np.zeros(system.universe.atoms.n_atoms,
                                   dtype=np.float32)
    testing.assert_allclose(system.universe.atoms.charges, charges,
                            err_msg="Charges should be 0.")


def test_nucleic6_masses():
    aa_universe: mda.Universe = mda.Universe(TPR, XTC)
    system: nucleic.Nucleic6 = nucleic.Nucleic6()
    system.create_topology(aa_universe)
    sel = system.universe.select_atoms("nucleic and name H1 H2 H3")

    c_mass: float = aa_universe.select_atoms("nucleic and name C4'").masses[0]
    masses: np.ndarray = np.repeat(c_mass, sel.n_atoms)
    testing.assert_allclose(sel.masses, masses,
                            err_msg=f"Masses should be {c_mass}.")
