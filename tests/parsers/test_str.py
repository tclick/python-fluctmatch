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
#   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS “AS IS”
#   AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
#   IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
#   ARE DISCLAIMED. IN NO EVENT SHALL THE REGENTS OR CONTRIBUTORS BE LIABLE FOR
#   ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
#   DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
#   SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
#   CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
#   OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
#   OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
#   Timothy H. Click, Nixon Raj, and Jhih-Wei Chu.
#   Simulation. Meth Enzymology. 578 (2016), 327-342,
#   Calculation of Enzyme Fluctuograms from All-Atom Molecular Dynamics
#   doi:10.1016/bs.mie.2016.05.024.
#
# ------------------------------------------------------------------------------

from pathlib import Path
from typing import Union
from unittest.mock import patch

import MDAnalysis as mda
import pytest

import fluctmatch.parsers.writers.STR

from ..datafiles import COR
from ..datafiles import PSF
from ..datafiles import STR


class TestSTRWriter(object):
    @pytest.fixture()
    def u(self) -> mda.Universe:
        return mda.Universe(PSF, COR)

    def test_writer(self, u: mda.Universe, tmp_path: Path):
        filename: Path = tmp_path / "temp.stream"
        with patch("fluctmatch.parsers.writers.STR.Writer.write") as writer, \
                mda.Writer(filename, n_atoms=u.atoms.n_atoms) as w:
            w.write(u.atoms)
            writer.assert_called()

    def test_roundtrip(self, u: mda.Universe, tmp_path: Path):
        # Write out a copy of the Universe, and compare this against the
        # original.  This is more rigorous than simply checking the coordinates
        # as it checks all formatting
        filename: Path = tmp_path / "temp.stream"
        with mda.Writer(filename, n_atoms=u.atoms.n_atoms) as w:
            w.write(u.atoms)

        def STR_iter(fn: Union[str, Path]):
            with open(fn) as inf:
                for line in inf:
                    if not line.startswith('*'):
                        yield line

        for ref, other in zip(STR_iter(STR), STR_iter(filename)):
            assert ref == other
