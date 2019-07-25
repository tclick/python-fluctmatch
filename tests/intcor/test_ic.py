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

from pathlib import Path

import MDAnalysis as mda
import pandas as pd
import pytest
from numpy.testing import assert_allclose
from tests.datafiles import IC

from fluctmatch.intcor.IC import IntcorReader


class TestICWriter:
    @pytest.fixture()
    def u(self) -> pd.DataFrame:
        return IntcorReader(IC).read()

    @pytest.fixture()
    def outfile(self, tmpdir: str) -> Path:
        return Path(tmpdir) / "out.ic"

    def test_bond_distances(self, u: pd.DataFrame, outfile: Path):
        with mda.Writer(outfile) as ofile:
            ofile.write(u)

        u2 = IntcorReader(outfile).read()
        assert_allclose(u["r_IJ"], u2["r_IJ"],
                        err_msg="The distances don't match.")

    def test_roundtrip(self, u: pd.DataFrame, outfile: Path):
        # Write out a copy of the internal coordinates, and compare this against
        # the original. This is more rigorous as it checks all formatting.
        with mda.Writer(outfile) as ofile:
            ofile.write(u)

        def IC_iter(fn: str):
            with open(fn) as inf:
                for line in inf:
                    if not line.startswith('*'):
                        yield line

        for ref, other in zip(IC_iter(IC), IC_iter(outfile)):
            assert ref == other
