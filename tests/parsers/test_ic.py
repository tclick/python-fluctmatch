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
from unittest.mock import patch

import MDAnalysis as mda
import pytest
import static_frame as sf
from numpy.testing import assert_allclose

import fluctmatch.parsers.readers.IC as ICReader
from tests.datafiles import IC


class TestICReader:
    expected_rows: int = 7566
    expected_cols: int = 18

    @pytest.fixture()
    def u(self) -> sf.Frame:
        return ICReader.Reader(IC).read()

    def test_reader(self, u: sf.Frame):
        rows, cols = u.shape
        assert self.expected_rows == rows
        assert self.expected_cols == cols


class TestICWriter:
    @pytest.fixture()
    def u(self) -> sf.Frame:
        return ICReader.Reader(IC).read()

    def test_writer(self, u: sf.Frame, tmp_path: Path) -> None:
        filename = tmp_path / "temp.ic"
        with patch("fluctmatch.parsers.writers.IC.Writer.write") as icw, mda.Writer(
            filename
        ) as ofile:
            ofile.write(u)
            icw.assert_called()

    def test_bond_distances(self, u: sf.Frame, tmp_path: Path) -> None:
        filename: Path = tmp_path / "temp.ic"
        with mda.Writer(filename) as ofile:
            ofile.write(u)

        u2 = ICReader.Reader(tmp_path / "temp.ic").read()
        assert_allclose(u["r_IJ"], u2["r_IJ"], err_msg="The distances don't match.")

    def test_roundtrip(self, u: sf.Frame, tmp_path: Path):
        # Write out a copy of the internal coordinates, and compare this against
        # the original. This is more rigorous as it checks all formatting.
        filename: Path = tmp_path / "temp.ic"
        with mda.Writer(filename) as ofile:
            ofile.write(u)

        def IC_iter(fn: str):
            with open(fn) as inf:
                for line in inf:
                    if not line.startswith("*"):
                        yield line

        for ref, other in zip(IC_iter(IC), IC_iter(filename)):
            assert ref == other
