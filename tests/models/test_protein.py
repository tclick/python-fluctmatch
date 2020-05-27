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
"""Tests for various protein core."""

from typing import List

import MDAnalysis as mda
import numpy as np
import pytest
from numpy import testing

from fluctmatch.core.models import calpha, caside, ncsc, polar

from ..datafiles import TPR, XTC


class TestCalpha:
    @pytest.fixture(scope="class")
    def u(self) -> mda.Universe:
        universe = mda.Universe(TPR, XTC)
        return mda.Merge(universe.residues[:6].atoms)

    @pytest.fixture(scope="class")
    def system(self) -> calpha.Model:
        return calpha.Model()

    def test_creation(self, u: mda.Universe, system: calpha.Model):
        system.create_topology(u)

        n_atoms = u.select_atoms("calpha or bioion").n_atoms
        testing.assert_equal(
            system.universe.atoms.n_atoms,
            n_atoms,
            err_msg="Number of sites don't match.",
        )

    def test_positions(self, u: mda.Universe, system: calpha.Model):
        cg_universe: mda.Universe = system.transform(u)

        positions: np.ndarray = np.asarray(
            [
                residue.atoms.select_atoms(sel).center_of_mass()
                for residue in u.select_atoms("protein or bioion").residues
                for sel in system._mapping.values()
                if residue.atoms.select_atoms(sel)
            ]
        )

        testing.assert_allclose(
            cg_universe.atoms.positions,
            positions,
            err_msg="The coordinates do not match.",
        )

    def test_trajectory(self, u: mda.Universe, system: calpha.Model):
        cg_universe: mda.Universe = system.transform(u)

        testing.assert_equal(
            cg_universe.trajectory.n_frames,
            u.trajectory.n_frames,
            err_msg="All-atom and coarse-grain trajectories unequal.",
        )


class TestCaside:
    @pytest.fixture(scope="class")
    def u(self) -> mda.Universe:
        universe = mda.Universe(TPR, XTC)
        return mda.Merge(universe.residues[:6].atoms)

    @pytest.fixture(scope="class")
    def system(self) -> caside.Model:
        return caside.Model()

    def test_creation(self, u: mda.Universe, system: caside.Model):
        system.create_topology(u)

        n_atoms = u.select_atoms("calpha or cbeta or bioion").n_atoms
        testing.assert_equal(
            system.universe.atoms.n_atoms,
            n_atoms,
            err_msg="Number of sites not equal.",
        )

    def test_positions(self, u: mda.Universe, system: caside.Model):
        cg_universe: mda.Universe = system.transform(u)

        positions: np.ndarray = np.asarray(
            [
                residue.atoms.select_atoms(sel).center_of_mass()
                for residue in u.select_atoms("protein or bioion").residues
                for sel in system._mapping.values()
                if residue.atoms.select_atoms(sel)
            ]
        )

        testing.assert_allclose(
            cg_universe.atoms.positions,
            positions,
            err_msg="The coordinates do not match.",
        )

    def test_trajectory(self, u: mda.Universe, system: caside.Model):
        cg_universe: mda.Universe = system.transform(u)

        testing.assert_equal(
            cg_universe.trajectory.n_frames,
            u.trajectory.n_frames,
            err_msg="All-atom and coarse-grain trajectories unequal.",
        )


class TestNcsc:
    @pytest.fixture(scope="class")
    def u(self) -> mda.Universe:
        universe = mda.Universe(TPR, XTC)
        return mda.Merge(universe.residues[:6].atoms)

    @pytest.fixture(scope="class")
    def system(self) -> ncsc.Model:
        return ncsc.Model()

    def test_creation(self, u: mda.Universe, system: ncsc.Model):
        system.create_topology(u)
        n_atoms = u.select_atoms(
            "(protein and name N O OT1) or cbeta or bioion"
        ).n_atoms
        testing.assert_equal(
            system.universe.atoms.n_atoms,
            n_atoms,
            err_msg="Number of sites not equal.",
        )

    def test_positions(self, u: mda.Universe, system: ncsc.Model):
        cg_universe: mda.Universe = system.transform(u)

        positions: np.ndarray = np.asarray(
            [
                residue.atoms.select_atoms(sel).center_of_mass()
                for residue in u.select_atoms("protein or bioion").residues
                for sel in system._mapping.values()
                if residue.atoms.select_atoms(sel)
            ]
        )

        testing.assert_allclose(
            cg_universe.atoms.positions,
            positions,
            err_msg="The coordinates do not match.",
        )

    def test_trajectory(self, u: mda.Universe, system: ncsc.Model):
        cg_universe: mda.Universe = system.transform(u)

        testing.assert_equal(
            cg_universe.trajectory.n_frames,
            u.trajectory.n_frames,
            err_msg="All-atom and coarse-grain trajectories unequal.",
        )


class TestPolar:
    @pytest.fixture(scope="class")
    def u(self) -> mda.Universe:
        universe = mda.Universe(TPR, XTC)
        return mda.Merge(universe.residues[:6].atoms)

    @pytest.fixture(scope="class")
    def system(self) -> polar.Model:
        return polar.Model()

    @pytest.fixture(scope="class")
    def other(self) -> ncsc.Model:
        return ncsc.Model()

    def test_creation(self, u: mda.Universe, system: polar.Model):
        system.create_topology(u)
        n_atoms = u.select_atoms(
            "(protein and name N O OT1) or cbeta or bioion"
        ).n_atoms
        testing.assert_equal(
            system.universe.atoms.n_atoms,
            n_atoms,
            err_msg="Number of sites not equal.",
        )

    def test_positions(self, u: mda.Universe, system: polar.Model):
        cg_universe: mda.Universe = system.transform(u)
        beads: List[mda.AtomGroup] = []

        for residue in u.select_atoms("protein or bioion").residues:
            for sel in system._mapping.values():
                if isinstance(sel, dict):
                    value: mda.AtomGroup = sel.get(
                        residue.resname, "hsidechain and not name H*"
                    )
                    bead: mda.AtomGroup = residue.atoms.select_atoms(value)
                else:
                    bead: mda.AtomGroup = residue.atoms.select_atoms(sel)
                if bead:
                    beads.append(bead)

        positions: np.ndarray = np.asarray(
            [bead.center_of_mass() for bead in beads if bead]
        )

        testing.assert_allclose(
            cg_universe.atoms.positions,
            positions,
            err_msg="The coordinates do not match.",
        )

    def test_trajectory(self, u: mda.Universe, system: polar.Model):
        cg_universe: mda.Universe = system.transform(u)

        testing.assert_equal(
            cg_universe.trajectory.n_frames,
            u.trajectory.n_frames,
            err_msg="All-atom and coarse-grain trajectories unequal.",
        )

    def test_ncsc_polar_positions(
        self, u: mda.Universe, system: polar.Model, other: ncsc.Model
    ):
        polar_universe: mda.Universe = system.transform(u)

        ncsc_universe: mda.Universe = other.transform(u)

        testing.assert_raises(
            AssertionError,
            testing.assert_allclose,
            polar_universe.atoms.positions,
            ncsc_universe.atoms.positions,
        )
