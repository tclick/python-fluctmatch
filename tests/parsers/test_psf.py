# ------------------------------------------------------------------------------
#   python-fluctmatch
#   Copyright (c) 2013-2020 Timothy H. Click, Ph.D.
#
#   All rights reserved.
#
#   Redistribution and use in source and binary forms, with or without
#   modification, are permitted provided that the following conditions are met:
#
#   Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
#   Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
#   Neither the name of the author nor the names of its contributors may be used
#   to endorse or promote products derived from this software without specific
#   prior written permission.
#
#    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS “AS IS”
#    AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
#    IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
#    ARE DISCLAIMED. IN NO EVENT SHALL THE REGENTS OR CONTRIBUTORS BE LIABLE FOR
#    ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
#    DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
#    SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
#    CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
#    LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
#    OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH
#    DAMAGE.
#
#   Timothy H. Click, Nixon Raj, and Jhih-Wei Chu.
#   Simulation. Meth Enzymology. 578 (2016), 327-342,
#   Calculation of Enzyme Fluctuograms from All-Atom Molecular Dynamics
#   doi:10.1016/bs.mie.2016.05.024.
#
# ------------------------------------------------------------------------------

from pathlib import Path
from typing import ClassVar
from typing import List
from unittest.mock import patch

import MDAnalysis as mda
import pytest
from MDAnalysis.core.topologyobjects import TopologyObject
from MDAnalysisTests.topology.base import ParserBase
from numpy.testing import assert_equal

import fluctmatch.parsers.parsers.PSF as PSFParser
import fluctmatch.parsers.writers.PSF

from ..datafiles import COR
from ..datafiles import PSF


class TestPSFWriter(object):
    @pytest.fixture(scope="class")
    def u(self) -> mda.Universe:
        return mda.Universe(PSF, COR)

    def test_writer(self, u: mda.Universe, tmp_path: Path):
        filename: Path = tmp_path / "temp.xplor.psf"
        with patch("fluctmatch.parsers.writers.PSF.Writer.write") as writer, \
                mda.Writer(filename) as w:
            w.write(u.atoms)
            writer.assert_called()

    def test_roundtrip(self, u: mda.Universe, tmp_path: Path):
        # Write out a copy of the Universe, and compare this against the
        # original. This is more rigorous than simply checking the coordinates
        # as it checks all formatting
        filename: Path = tmp_path / "temp.xplor.psf"
        with mda.Writer(filename) as w:
            w.write(u.atoms)

        def PSF_iter(fn: str):
            with open(fn) as inf:
                for line in inf:
                    if not line.startswith('*'):
                        yield line

        for ref, other in zip(PSF_iter(PSF), PSF_iter(filename.as_posix())):
            assert ref == other

    def test_write_atoms(self, u: mda.Universe, tmp_path: Path):
        # Test that written file when read gives same coordinates
        filename: Path = tmp_path / "temp.xplor.psf"
        with mda.Writer(filename) as w:
            w.write(u.atoms)

        u2: mda.Universe = mda.Universe(filename, COR)

        assert_equal(u.atoms.charges, u2.atoms.charges)


class TestPSFParser(ParserBase):
    """
    Based on small PDB with AdK (:data:`PDB_small`).
    """
    parser: ClassVar[PSFParser.Reader] = PSFParser.Reader
    ref_filename: ClassVar[str] = PSF
    expected_attrs: ClassVar[List[str]] = ["ids", "names", "masses",
                                           "charges", "resids", "resnames",
                                           "segids", "bonds", "angles",
                                           "dihedrals", "impropers"]
    expected_n_atoms: ClassVar[int] = 330
    expected_n_residues: ClassVar[int] = 115
    expected_n_segments: ClassVar[int] = 1

    def test_bonds_total_counts(self, top: TopologyObject):
        assert len(top.bonds.values) == 429

    def test_bonds_atom_counts(self, filename: str):
        u = mda.Universe(filename)
        assert len(u.atoms[[0]].bonds) == 2
        assert len(u.atoms[[42]].bonds) == 2

    def test_bonds_identity(self, top: TopologyObject):
        vals = top.bonds.values
        for b in ((0, 1), (0, 2)):
            assert (b in vals) or (b[::-1] in vals)

    def test_angles_total_counts(self, top: TopologyObject):
        assert len(top.angles.values) == 726

    def test_angles_atom_counts(self, filename: str):
        u = mda.Universe(filename)
        assert len(u.atoms[[0]].angles), 4
        assert len(u.atoms[[42]].angles), 6

    def test_angles_identity(self, top: TopologyObject):
        vals = top.angles.values
        for b in ((1, 0, 2), (0, 1, 2), (0, 2, 3)):
            assert (b in vals) or (b[::-1] in vals)

    def test_dihedrals_total_counts(self, top: TopologyObject):
        assert len(top.dihedrals.values) == 907

    def test_dihedrals_atom_counts(self, filename: str):
        u = mda.Universe(filename)
        assert len(u.atoms[[0]].dihedrals) == 4

    def test_dihedrals_identity(self, top: TopologyObject):
        vals = top.dihedrals.values
        for b in ((0, 1, 2, 3), (0, 2, 3, 4), (0, 2, 3, 5), (1, 0, 2, 3)):
            assert (b in vals) or (b[::-1] in vals)
