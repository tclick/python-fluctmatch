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
"""Tests for different solvent core."""

import MDAnalysis as mda
import numpy as np
import pytest
from numpy import testing

from fluctmatch.core.models import dma, tip3p, water

from ..datafiles import DMA, TIP3P


class TestWater:
    @pytest.fixture(scope="class")
    def u(self) -> mda.Universe:
        return mda.Universe(TIP3P)

    @pytest.fixture(scope="class")
    def system(self) -> water.Model:
        return water.Model()

    def test_creation(self, u: mda.Universe, system: water.Model):
        system.create_topology(u)

        n_atoms: int = sum(
            u.select_atoms(select).residues.n_residues
            for select in system._mapping.values()
        )

        testing.assert_equal(
            system.universe.atoms.n_atoms,
            n_atoms,
            err_msg="Number of sites don't match.",
        )

    def test_positions(self, u: mda.Universe, system: water.Model):
        cg_universe: mda.Universe = system.transform(u)

        positions: np.ndarray = np.asarray(
            [
                _.atoms.select_atoms(select).center_of_mass()
                for _ in u.select_atoms("water").residues
                for select in system._mapping.values()
                if _.atoms.select_atoms(select)
            ]
        )

        testing.assert_allclose(
            np.asarray(positions),
            cg_universe.atoms.positions,
            err_msg="The coordinates do not match.",
        )

    def test_bonds(self, u: mda.Universe, system: water.Model):
        aa_universe: mda.Universe = mda.Universe(TIP3P)
        system: water.Model = water.Model()
        cg_universe: mda.Universe = system.transform(aa_universe)

        testing.assert_equal(
            len(cg_universe.bonds), 0, err_msg="No bonds should exist."
        )

    def test_mass(self, u: mda.Universe, system: water.Model):
        cg_universe: mda.Universe = system.transform(u)

        masses: np.ndarray = np.fromiter(
            [
                _.atoms.select_atoms(select).total_mass()
                for _ in u.select_atoms("water").residues
                for select in system._selection.values()
                if _.atoms.select_atoms(select)
            ],
            dtype=np.float32,
        )

        testing.assert_allclose(
            cg_universe.atoms.masses, masses, err_msg="The masses do not match."
        )

    def test_charges(self, u: mda.Universe, system: water.Model):
        cg_universe: mda.Universe = system.transform(u)

        testing.assert_allclose(
            cg_universe.atoms.charges,
            np.zeros(cg_universe.atoms.n_atoms),
            err_msg="The masses do not match.",
        )

    def test_creation_from_tip4p(self, u: mda.Universe, system: water.Model):
        system.create_topology(u)

        n_atoms: int = sum(
            u.select_atoms(select).residues.n_residues
            for select in system._mapping.values()
        )

        testing.assert_equal(
            system.universe.atoms.n_atoms,
            n_atoms,
            err_msg="Number of sites don't match.",
        )

    def test_positions_from_tip4p(self, u: mda.Universe, system: water.Model):
        cg_universe: mda.Universe = system.transform(u)

        positions: np.ndarray = np.asarray(
            [
                _.atoms.select_atoms(select).center_of_mass()
                for _ in u.select_atoms("water").residues
                for select in system._mapping.values()
                if _.atoms.select_atoms(select)
            ]
        )

        testing.assert_allclose(
            np.asarray(positions),
            cg_universe.atoms.positions,
            err_msg="The coordinates do not match.",
        )

    def test_bonds_from_tip4p(self, u: mda.Universe, system: water.Model):
        cg_universe: mda.Universe = system.transform(u)

        testing.assert_equal(
            len(cg_universe.bonds), 0, err_msg="No bonds should exist."
        )


class TestTip3p:
    @pytest.fixture(scope="class")
    def u(self) -> mda.Universe:
        return mda.Universe(TIP3P)

    @pytest.fixture(scope="class")
    def system(self) -> tip3p.Model:
        return tip3p.Model()

    def test_creation(self, u: mda.Universe, system: tip3p.Model):
        system.create_topology(u)

        n_atoms: int = sum(
            u.select_atoms(select).residues.n_residues
            for select in system._mapping.values()
        )

        testing.assert_equal(
            system.universe.atoms.n_atoms,
            n_atoms,
            err_msg="Number of sites don't match.",
        )

    def test_tip3p_positions(self, u: mda.Universe, system: tip3p.Model):
        cg_universe: mda.Universe = system.transform(u)

        positions: np.ndarray = np.asarray(
            [
                _.atoms.select_atoms(select).center_of_mass()
                for _ in u.select_atoms("water").residues
                for select in system._mapping.values()
                if _.atoms.select_atoms(select)
            ]
        )

        testing.assert_allclose(
            np.asarray(positions),
            cg_universe.atoms.positions,
            err_msg="The coordinates do not match.",
        )

    def test_tip3p_bonds(self, u: mda.Universe, system: tip3p.Model):
        cg_universe: mda.Universe = system.transform(u)

        testing.assert_equal(
            len(cg_universe.bonds),
            system.universe.residues.n_residues * 3,
            err_msg=("Expected and actual number of bonds " "not equal"),
        )
        testing.assert_equal(
            len(cg_universe.angles),
            system.universe.residues.n_residues * 3,
            err_msg=("Expected and actual number of angles " "not equal"),
        )
        testing.assert_equal(
            len(cg_universe.dihedrals), 0, err_msg="No dihedral angles should exist.",
        )
        testing.assert_equal(
            len(cg_universe.impropers),
            0,
            err_msg=("No improper dihedral angles " "should exist."),
        )


class TestDma:
    @pytest.fixture(scope="class")
    def u(self) -> mda.Universe:
        return mda.Universe(DMA)

    @pytest.fixture()
    def system(self) -> dma.Model:
        return dma.Model()

    def test_creation(self, u: mda.Universe, system: dma.Model):
        system.create_topology(u)

        n_atoms: int = sum(
            u.select_atoms(select).residues.n_residues
            for select in system._mapping.values()
        )

        testing.assert_equal(
            system.universe.atoms.n_atoms,
            n_atoms,
            err_msg="Number of sites don't match.",
        )

    def test_positions(self, u: mda.Universe, system: dma.Model):
        cg_universe: mda.Universe = system.transform(u)

        positions: np.ndarray = np.asarray(
            [
                _.atoms.select_atoms(select).center_of_mass()
                for _ in u.select_atoms("resname DMA").residues
                for select in system._mapping.values()
                if _.atoms.select_atoms(select)
            ]
        )

        testing.assert_allclose(
            positions,
            cg_universe.atoms.positions,
            err_msg="The coordinates do not match.",
        )

    def test_bonds(self, u: mda.Universe, system: dma.Model):
        cg_universe: mda.Universe = system.transform(u)

        testing.assert_equal(
            len(cg_universe.bonds),
            system.universe.residues.n_residues * 3,
            err_msg=("Expected and actual number of bonds " "not equal"),
        )
        testing.assert_equal(
            len(cg_universe.angles),
            system.universe.residues.n_residues * 3,
            err_msg=("Expected and actual number of angles " "not equal"),
        )
        testing.assert_equal(
            len(cg_universe.dihedrals), 0, err_msg="No dihedral angles should exist.",
        )
        testing.assert_equal(
            len(cg_universe.impropers),
            system.universe.residues.n_residues * 3,
            err_msg=(
                "Expected and actual number of improper " "dihedral angles not equal."
            ),
        )
