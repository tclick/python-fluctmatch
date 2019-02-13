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
"""Tests for the elastic network model."""

import MDAnalysis as mda
import pytest
from numpy import testing

from fluctmatch.models import enm
from ..datafiles import CG_PSF, CG_DCD


class TestEnm:
    @pytest.fixture(scope="class")
    def u(self) -> mda.Universe:
        return mda.Universe(CG_PSF, CG_DCD)

    @pytest.fixture(scope="class")
    def system(self) -> enm.Enm:
        return enm.Enm()
    
    def test_creation(self, u: mda.Universe, system: enm.Enm):
        cg_universe: mda.Universe = system.transform(u)
        n_atoms: int = u.atoms.n_atoms
    
        testing.assert_equal(cg_universe.atoms.n_atoms, n_atoms,
                             err_msg="The number of beads don't match.")
    
    def test_names(self, u: mda.Universe, system: enm.Enm):
        cg_universe: mda.Universe = system.transform(u)
    
        testing.assert_string_equal(cg_universe.atoms[0].name, "A001")
        testing.assert_string_equal(cg_universe.residues[0].resname, "A001")
    
    def test_positions(self, u: mda.Universe, system: enm.Enm):
        cg_universe = system.transform(u)
    
        testing.assert_allclose(cg_universe.atoms.positions, u.atoms.positions,
                                err_msg="Coordinates don't match.")
    
    def test_bonds(self, u: mda.Universe, system: enm.Enm):
        cg_universe = system.transform(u)
    
        assert len(cg_universe.bonds) > len(u.bonds)
