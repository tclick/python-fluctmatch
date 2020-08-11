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
"""Test additional MDAnalysis selection options."""

import MDAnalysis as mda
import pytest
from numpy import testing

from ..datafiles import GRO


class TestProteinSelections:
    @pytest.fixture(scope="class")
    def universe(self) -> mda.Universe:
        return mda.Universe(GRO)

    def test_backbone(self, universe: mda.Universe):
        sel = universe.select_atoms("backbone")
        testing.assert_equal(sel.n_atoms, 1890, "Number of atoms don't match.")

    def test_hbackbone(self, universe: mda.Universe):
        sel = universe.select_atoms("hbackbone")
        testing.assert_equal(sel.n_atoms, 2832, "Number of atoms don't match.")

    def test_calpha(self, universe):
        sel = universe.select_atoms("calpha")
        testing.assert_equal(sel.n_atoms, 472, "Number of atoms don't match.")

    def test_hcalpha(self, universe):
        sel = universe.select_atoms("hcalpha")
        testing.assert_equal(sel.n_atoms, 974, "Number of atoms don't match.")

    def test_cbeta(self, universe):
        sel = universe.select_atoms("cbeta")
        testing.assert_equal(sel.n_atoms, 442, "Number of atoms don't match.")

    def test_amine(self, universe):
        sel = universe.select_atoms("amine")
        testing.assert_equal(sel.n_atoms, 912, "Number of atoms don't match.")

    def test_carboxyl(self, universe):
        sel = universe.select_atoms("carboxyl")
        testing.assert_equal(sel.n_atoms, 946, "Number of atoms don't match.")

    def test_hsidechain(self, universe):
        sel = universe.select_atoms("hsidechain")
        testing.assert_equal(sel.n_atoms, 4374, "Number of atoms don't match.")


class TestSolvent:
    @pytest.fixture(scope="class")
    def universe(self) -> mda.Universe:
        return mda.Universe(GRO)

    def test_bioions(self, universe: mda.Universe):
        sel = universe.select_atoms("bioion")
        testing.assert_equal(sel.n_atoms, 4, "Number of atoms don't match.")

    def test_water(self, universe: mda.Universe):
        sel = universe.select_atoms("water")
        testing.assert_equal(sel.n_atoms, 72897, "Number of atoms don't match.")


class TestNucleic:
    @pytest.fixture(scope="class")
    def universe(self) -> mda.Universe:
        return mda.Universe(GRO)

    def test_nucleic(self, universe: mda.Universe):
        sel = universe.select_atoms("nucleic")
        testing.assert_equal(sel.n_atoms, 66, "Number of atoms don't match.")

    def test_hsugar(self, universe: mda.Universe):
        sel = universe.select_atoms("hnucleicsugar")
        testing.assert_equal(sel.n_atoms, 24, "Number of atoms don't match.")

    def test_hbase(self, universe: mda.Universe):
        sel = universe.select_atoms("hnucleicbase")
        testing.assert_equal(sel.n_atoms, 24, "Number of atoms don't match.")

    def test_hphosphate(self, universe: mda.Universe):
        sel = universe.select_atoms("nucleicphosphate")
        testing.assert_equal(sel.n_atoms, 18, "Number of atoms don't match.")

    def test_sugarc2(self, universe: mda.Universe):
        sel = universe.select_atoms("sugarC2")
        testing.assert_equal(sel.n_atoms, 10, "Number of atoms don't match.")

    def test_sugarc4(self, universe: mda.Universe):
        sel = universe.select_atoms("sugarC4")
        testing.assert_equal(sel.n_atoms, 14, "Number of atoms don't match.")

    def test_center(self, universe: mda.Universe):
        sel = universe.select_atoms("nucleiccenter")
        testing.assert_equal(sel.n_atoms, 4, "Number of atoms don't match.")
