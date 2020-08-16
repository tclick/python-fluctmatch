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

import itertools
from typing import List

import MDAnalysis as mda
import numpy as np
import pytest
from numpy import testing

from fluctmatch.core.models import dma, tip3p, water
from ..datafiles import DMA, TIP3P


class TestWater:
    @pytest.fixture(scope="class")
    def universe(self) -> mda.Universe:
        return mda.Universe(TIP3P)

    @pytest.fixture(scope="class")
    def model(self, universe: mda.Universe) -> water.Model:
        return water.Model()

    @pytest.fixture(scope="class")
    def system(self, universe: mda.Universe, model: water.Model) -> mda.Universe:
        return model.transform(universe)

    def test_creation(
        self, universe: mda.Universe, model: water.Model, system: mda.Universe
    ) -> None:
        n_atoms = 0
        for residue, selection in itertools.product(universe.residues, model._mapping):
            value = (
                selection.get(residue.resname)
                if isinstance(selection, dict)
                else selection
            )
            n_atoms += residue.atoms.select_atoms(value).residues.n_residues
        assert system.atoms.n_atoms == n_atoms, "Number of sites don't match."

    def test_positions(
        self, universe: mda.Universe, system: mda.Universe, model: water.Model
    ) -> None:
        positions: List[List[np.ndarray]] = []
        for residue, selection in itertools.product(universe.residues, model._mapping):
            value = (
                selection.get(residue.resname, "hsidechain and not name H*")
                if isinstance(selection, dict)
                else selection
            )
            if residue.atoms.select_atoms(value):
                positions.append(residue.atoms.select_atoms(value).center_of_mass())
        testing.assert_allclose(
            positions, system.atoms.positions, err_msg="The coordinates do not match.",
        )

    def test_masses(
        self, universe: mda.Universe, system: mda.Universe, model: water.Model
    ) -> None:
        masses = [
            residue.atoms.select_atoms(selection).total_mass()
            for residue, selection in itertools.product(
                universe.residues, model._selection
            )
            if residue.atoms.select_atoms(selection)
        ]
        testing.assert_allclose(
            system.atoms.masses, masses, err_msg="The masses do not match."
        )

    def test_charges(
        self, universe: mda.Universe, system: mda.Universe, model: water.Model
    ) -> None:
        try:
            charges = [
                residue.atoms.select_atoms(selection).total_charge()
                for residue, selection in itertools.product(
                    universe.residues, model._selection
                )
                if residue.atoms.select_atoms(selection)
            ]
        except mda.NoDataError:
            charges = [0.0] * system.atoms.n_atoms
        testing.assert_allclose(
            system.atoms.charges, charges, err_msg="The charges do not match.",
        )

    def test_bonds(self, system: mda.Universe) -> None:
        assert len(system.bonds) == 0, "Number of bonds should not be > 0."

    def test_angles(self, system: mda.Universe) -> None:
        with pytest.raises(mda.NoDataError) as error_info:
            system.angles

    def test_dihedrals(self, system: mda.Universe) -> None:
        with pytest.raises(mda.NoDataError) as error_info:
            system.dihedrals

    def test_impropers(self, system: mda.Universe) -> None:
        with pytest.raises(mda.NoDataError) as error_info:
            system.impropers


class TestTip3p(TestWater):
    @pytest.fixture(scope="class")
    def model(self, universe: mda.Universe) -> tip3p.Model:
        return tip3p.Model(guess_angles=True)

    def test_bonds(self, system: mda.Universe) -> None:
        assert len(system.bonds) > 0, "Number of bonds should be > 0."

    def test_angles(self, system: mda.Universe) -> None:
        assert len(system.angles) > 0, "Number of angles should be > 0."

    def test_dihedrals(self, system: mda.Universe) -> None:
        assert (
            len(system.dihedrals) == 0
        ), "Number of dihedral angles should not be > 0."

    def test_impropers(self, system: mda.Universe) -> None:
        assert (
            len(system.impropers) == 0
        ), "Number of improper angles should not be > 0."


class TestDma(TestTip3p):
    @pytest.fixture(scope="class")
    def universe(self) -> mda.Universe:
        return mda.Universe(DMA)

    @pytest.fixture(scope="class")
    def model(self) -> dma.Model:
        return dma.Model(guess_angles=True)

    def test_impropers(self, system: mda.Universe) -> None:
        assert len(system.impropers) > 0, "Number of improper angles should be > 0."
