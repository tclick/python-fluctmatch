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
from numpy import testing

from fluctmatch.models import enm
from ..datafiles import CG_PSF, CG_DCD


def test_enm_creation():
    aa_universe: mda.Universe = mda.Universe(CG_PSF, CG_DCD)
    system: enm.Enm = enm.Enm()
    cg_universe: mda.Universe = system.transform(aa_universe)
    n_atoms: int = aa_universe.atoms.n_atoms

    testing.assert_equal(cg_universe.atoms.n_atoms, n_atoms,
                         err_msg="The number of beads don't match.")


def test_enm_names():
    aa_universe: mda.Universe = mda.Universe(CG_PSF, CG_DCD)
    system: enm.Enm = enm.Enm()
    cg_universe: mda.Universe = system.transform(aa_universe)

    testing.assert_string_equal(cg_universe.atoms[0].name, "A001")
    testing.assert_string_equal(cg_universe.residues[0].resname, "A001")


def test_enm_positions():
    aa_universe: mda.Universe = mda.Universe(CG_PSF, CG_DCD)
    system: enm.Enm = enm.Enm()
    cg_universe = system.transform(aa_universe)

    testing.assert_allclose(cg_universe.atoms.positions,
                            aa_universe.atoms.positions,
                            err_msg="Coordinates don't match.")


def test_enm_bonds():
    aa_universe: mda.Universe = mda.Universe(CG_PSF, CG_DCD)
    system: enm.Enm = enm.Enm()
    cg_universe = system.transform(aa_universe)

    assert len(cg_universe.bonds) > len(aa_universe.bonds)
