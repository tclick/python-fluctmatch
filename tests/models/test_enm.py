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
#  Timothy H. Click, Nixon Raj, and Jhih-Wei Chuniverse.
#  Simulation. Meth Enzymology. 578 (2016), 327-342,
#  Calculation of Enzyme Fluctuograms from All-Atom Molecular Dynamics
#  doi:10.1016/bs.mie.2016.05.024.
"""Tests for the elastic network model."""

import MDAnalysis as mda
import pytest
from numpy import testing

from fluctmatch.core.models import enm
from ..datafiles import DCD, PSF


class TestEnm:
    @pytest.fixture(scope="class")
    def universe(self) -> mda.Universe:
        universe = mda.Universe(PSF, DCD)
        return mda.Merge(universe.residues[:6].atoms)

    @pytest.fixture(scope="class")
    def model(self, universe: mda.Universe) -> enm.Model:
        return enm.Model(charges=False)

    @pytest.fixture(scope="class")
    def system(self, universe: mda.Universe, model: enm.Model) -> mda.Universe:
        return model.transform(universe)

    def test_creation(self, universe: mda.Universe, system: mda.Universe) -> None:
        n_atoms: int = universe.atoms.n_atoms
        assert system.atoms.n_atoms == n_atoms, "The number of beads don't match."

    def test_names(self, universe: mda.Universe, system: mda.Universe) -> None:
        testing.assert_string_equal(system.atoms[0].name, "A001")
        testing.assert_string_equal(system.residues[0].resname, "A001")

    def test_positions(self, universe: mda.Universe, system: mda.Universe) -> None:
        testing.assert_allclose(
            system.atoms.positions,
            universe.atoms.positions,
            err_msg="Coordinates don't match.",
        )

    def test_trajectory(self, universe: mda.Universe, system: mda.Universe) -> None:
        assert (
            system.trajectory.n_frames == universe.trajectory.n_frames
        ), "All-atom and coarse-grain trajectories unequal."

    def test_bonds(self, universe: mda.Universe, system: mda.Universe) -> None:
        assert len(system.bonds) > len(
            universe.bonds
        ), "# of ENM bonds should be greater than the # of original CG bonds."

    def test_angles(self, system: mda.Universe) -> None:
        assert len(system.angles) == 0, "Number of angles should not be > 0."

    def test_dihedrals(self, system: mda.Universe) -> None:
        assert (
            len(system.dihedrals) == 0
        ), "Number of dihedral angles should not be > 0."

    def test_impropers(self, system: mda.Universe) -> None:
        assert (
            len(system.impropers) == 0
        ), "Number of improper angles should not not be > 0."
