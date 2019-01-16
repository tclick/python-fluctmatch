# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding: utf-8 -*-
#
#  python-fluctmatch - Fluctuation matching library for Python
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
#  Please cite your use of fluctmatch in published work:
#
#  Timothy H. Click, Nixon Raj, and Jhih-Wei Chu.
#  Calculation of Enzyme Fluctuograms from All-Atom Molecular Dynamics
#  Simulation. Meth Enzymology. 578 (2016), 327-342,
#  doi:10.1016/bs.mie.2016.05.024.

import MDAnalysis as mda
from numpy import testing
from fluctmatch.models import ions
from fluctmatch.models.selection import *
from ..datafiles import IONS


def test_ions_creation():
    aa_universe = mda.Universe(IONS)
    cg_universe = ions.SolventIons(IONS)
    cg_natoms = (
        aa_universe.select_atoms("name LI LIT K NA F CL BR I").n_atoms)
    testing.assert_equal(
        cg_universe.atoms.n_atoms,
        cg_natoms,
        err_msg="Number of sites don't match.",
        verbose=True,
    )


def test_ions_positions():
    positions = []
    aa_universe = mda.Universe(IONS)
    cg_universe = ions.SolventIons(IONS)
    for _ in aa_universe.select_atoms("name LI LIT K NA F CL BR I").residues:
        positions.append(
            _.atoms.select_atoms("name LI LIT K NA F CL BR I")
            .center_of_mass())
    testing.assert_allclose(
        np.array(positions),
        cg_universe.atoms.positions,
        err_msg="The coordinates do not match.",
    )


