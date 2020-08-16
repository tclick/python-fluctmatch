# fluctmatch --- https://github.com/tclick/python-fluctmatch
# Copyright (c) 2013-2020 The fluctmatch Development Team and contributors
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
#

import itertools
from typing import List

import MDAnalysis as mda
import numpy as np
import pytest
from numpy import testing

from fluctmatch.core.models import united
from ..datafiles import TPR, XTC

# Number of residues to test
N_RESIDUES = 6


class TestUnited:
    @pytest.fixture(scope="class")
    def universe(self) -> mda.Universe:
        return mda.Universe(TPR, XTC)

    @pytest.fixture(scope="class")
    def model(self) -> united.Model:
        return united.Model(guess_angles=True)

    @pytest.fixture(scope="class")
    def system(self, universe: mda.Universe, model: united.Model) -> mda.Universe:
        return model.transform(universe)

    def test_creation(
        self, universe: mda.Universe, model: united.Model, system: mda.Universe,
    ) -> None:
        n_atoms = 0
        for residue, selection in itertools.product(universe.residues, model._mapping):
            value = (
                selection.get(residue.resname)
                if isinstance(selection, dict)
                else selection
            )
            n_atoms += residue.atoms.select_atoms(value).n_atoms
        testing.assert_equal(
            system.atoms.n_atoms, n_atoms, err_msg="Number of sites don't match.",
        )

    def test_positions(
        self, universe: mda.Universe, system: mda.Universe, model: united.Model
    ) -> None:
        positions: List[np.ndarray] = []
        for residue, selection in itertools.product(universe.residues, model._mapping):
            value = (
                selection.get(residue.resname, "hsidechain and not name H*")
                if isinstance(selection, dict)
                else selection
            )
            if residue.atoms.select_atoms(value):
                positions.append(residue.atoms.select_atoms(value).positions)
        positon_array = np.concatenate(positions, axis=0)
        testing.assert_allclose(
            positon_array,
            system.atoms.positions,
            err_msg="The coordinates do not match.",
        )

    def test_masses(
        self, universe: mda.Universe, system: mda.Universe, model: united.Model
    ) -> None:
        masses = np.concatenate(
            [
                residue.atoms.select_atoms(selection).masses
                for residue, selection in itertools.product(
                    universe.residues, model._selection
                )
                if residue.atoms.select_atoms(selection)
            ]
        )
        testing.assert_allclose(
            system.atoms.masses, masses, err_msg="The masses do not match."
        )

    def test_charges(
        self, universe: mda.Universe, system: mda.Universe, model: united.Model
    ) -> None:
        try:
            charges = np.concatenate(
                [
                    residue.atoms.select_atoms(selection).charges
                    for residue, selection in itertools.product(
                        universe.residues, model._selection
                    )
                    if residue.atoms.select_atoms(selection)
                ]
            )
        except mda.NoDataError:
            charges = [0.0] * system.atoms.n_atoms
        testing.assert_allclose(
            system.atoms.charges, charges, err_msg="The charges do not match.",
        )

    def test_trajectory(self, universe: mda.Universe, system: mda.Universe) -> None:
        testing.assert_equal(
            system.trajectory.n_frames,
            universe.trajectory.n_frames,
            err_msg="All-atom and coarse-grain trajectories unequal.",
        )

    def test_bonds(self, system: mda.Universe) -> None:
        assert len(system.bonds) > 0, "Number of bonds should be > 0."

    def test_angles(self, system: mda.Universe) -> None:
        assert len(system.angles) > 0, "Number of angles should be > 0."

    def test_dihedrals(self, system: mda.Universe) -> None:
        assert len(system.dihedrals) > 0, "Number of dihedral angles should be > 0."

    def test_impropers(self, system: mda.Universe) -> None:
        assert len(system.impropers) > 0, "Number of improper angles should be > 0."
