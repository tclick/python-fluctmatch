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

import MDAnalysis as mda
from numpy import testing

from fluctmatch.models import protein
from fluctmatch.models.selection import *
from ..datafiles import (
    PDB_prot,
    TPR,
    XTC,
)


def test_calpha_creation():
    aa_universe: mda.Universe = mda.Universe(PDB_prot)
    system: protein.Calpha = protein.Calpha()
    system.create_topology(aa_universe)

    n_atoms: int = sum(aa_universe.select_atoms(selection).residues.n_residues
                       for selection in system._mapping.values())

    testing.assert_equal(system.universe.atoms.n_atoms, n_atoms,
                         err_msg="Number of sites don't match.", verbose=True)


def test_calpha_positions():
    aa_universe: mda.Universe = mda.Universe(PDB_prot)
    system: protein.Calpha = protein.Calpha()
    cg_universe: mda.Universe = system.transform(aa_universe)

    positions: np.ndarray = np.asarray([
        _.atoms.select_atoms(selection).center_of_mass()
        for _ in aa_universe.select_atoms("calpha or bioion").residues
        for selection in system._mapping.values()
        if _.atoms.select_atoms(selection)
    ])

    testing.assert_allclose(
        np.asarray(positions),
        cg_universe.atoms.positions,
        err_msg="The coordinates do not match.",
    )


def test_calpha_trajectory():
    aa_universe = mda.Universe(TPR, XTC)
    cg_universe = protein.Calpha(TPR, XTC)
    testing.assert_equal(
        cg_universe.trajectory.n_frames,
        aa_universe.trajectory.n_frames,
        err_msg=native_str("All-atom and coarse-grain trajectories unequal."),
        verbose=True,
    )


def test_caside_creation():
    aa_universe = mda.Universe(PDB_prot)
    cg_universe = protein.Caside(PDB_prot)
    cg_natoms = (aa_universe.select_atoms("calpha").n_atoms +
                 aa_universe.select_atoms("cbeta").n_atoms +
                 aa_universe.select_atoms("bioion").n_atoms)
    testing.assert_equal(
        cg_universe.atoms.n_atoms,
        cg_natoms,
        err_msg=native_str("Number of sites not equal."),
        verbose=True,
    )


def test_caside_positions():
    positions = []
    aa_universe = mda.Universe(PDB_prot)
    cg_universe = protein.Caside(PDB_prot)
    for _ in aa_universe.select_atoms("protein").residues:
        positions.append(_.atoms.select_atoms("calpha").center_of_mass())
        if _.resname != "GLY":
            cbeta = "hsidechain and not name H*"
            positions.append(_.atoms.select_atoms(cbeta).center_of_mass())
    for _ in aa_universe.select_atoms("bioion").residues:
        positions.append(_.atoms.center_of_mass())
    testing.assert_allclose(
        np.array(positions),
        cg_universe.atoms.positions,
        err_msg=native_str("The coordinates do not match."),
    )


def test_caside_trajectory():
    aa_universe = mda.Universe(TPR, XTC)
    cg_universe = protein.Caside(TPR, XTC)
    testing.assert_equal(
        cg_universe.trajectory.n_frames,
        aa_universe.trajectory.n_frames,
        err_msg=native_str("All-atom and coarse-grain trajectories unequal."),
        verbose=True,
    )


def test_ncsc_creation():
    aa_universe = mda.Universe(PDB_prot)
    cg_universe = protein.Ncsc(PDB_prot)
    cg_natoms = (aa_universe.select_atoms("protein and name N").n_atoms +
                 aa_universe.select_atoms("protein and name O OT1").n_atoms +
                 aa_universe.select_atoms("cbeta").n_atoms +
                 aa_universe.select_atoms("bioion").n_atoms)
    testing.assert_equal(
        cg_universe.atoms.n_atoms,
        cg_natoms,
        err_msg=native_str("Number of sites not equal."),
        verbose=True,
    )


def test_ncsc_positions():
    positions = []
    aa_universe = mda.Universe(PDB_prot)
    cg_universe = protein.Ncsc(PDB_prot)
    for _ in aa_universe.select_atoms("protein").residues:
        positions.append(_.atoms.select_atoms("name N").center_of_mass())
        if _.resname != "GLY":
            cbeta = "hsidechain and not name H*"
            positions.append(_.atoms.select_atoms(cbeta).center_of_mass())
        positions.append(
            _.atoms.select_atoms("name O OT1 OT2 OXT").center_of_mass())
    for _ in aa_universe.select_atoms("bioion").residues:
        positions.append(_.atoms.center_of_mass())
    testing.assert_allclose(
        np.array(positions),
        cg_universe.atoms.positions,
        err_msg=native_str("The coordinates do not match."),
    )


def test_ncsc_trajectory():
    aa_universe = mda.Universe(TPR, XTC)
    cg_universe = protein.Ncsc(TPR, XTC)
    testing.assert_equal(
        cg_universe.trajectory.n_frames,
        aa_universe.trajectory.n_frames,
        err_msg=native_str("All-atom and coarse-grain trajectories unequal."),
        verbose=True,
    )


def test_polar_creation():
    aa_universe = mda.Universe(PDB_prot)
    cg_universe = protein.Polar(PDB_prot)
    cg_natoms = (aa_universe.select_atoms("protein and name N").n_atoms +
                 aa_universe.select_atoms("protein and name O OT1").n_atoms +
                 aa_universe.select_atoms("cbeta").n_atoms +
                 aa_universe.select_atoms("bioion").n_atoms)
    testing.assert_equal(
        cg_universe.atoms.n_atoms,
        cg_natoms,
        err_msg=native_str("Number of sites not equal."),
        verbose=True,
    )


def test_polar_positions():
    positions = []
    aa_universe = mda.Universe(PDB_prot)
    cg_universe = protein.Polar(PDB_prot)
    for _ in aa_universe.select_atoms("protein").residues:
        positions.append(_.atoms.select_atoms("name N").center_of_mass())
        if _.resname != "GLY":
            cbeta = cg_universe._mapping["CB"].get(
                _.resname, "hsidechain and not name H*")
            positions.append(_.atoms.select_atoms(cbeta).center_of_mass())
        positions.append(
            _.atoms.select_atoms("name O OT1 OT2 OXT").center_of_mass())
    for _ in aa_universe.select_atoms("bioion").residues:
        positions.append(_.atoms.center_of_mass())
    testing.assert_allclose(
        np.array(positions),
        cg_universe.atoms.positions,
        err_msg=native_str("The coordinates do not match."),
    )


def test_polar_trajectory():
    aa_universe = mda.Universe(TPR, XTC)
    cg_universe = protein.Polar(TPR, XTC)
    testing.assert_equal(
        cg_universe.trajectory.n_frames,
        aa_universe.trajectory.n_frames,
        err_msg=native_str("All-atom and coarse-grain trajectories unequal."),
        verbose=True,
    )
