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
from typing import Dict
from typing import Mapping
from typing import Union

import MDAnalysis as mda
import pandas as pd
import pytest
from numpy.testing import assert_allclose
from tests.datafiles import PRM

from fluctmatch.parameter.PRM import ParamReader


class TestPRMWriter(object):
    @pytest.fixture()
    def u(self) -> Dict[str, pd.DataFrame]:
        return ParamReader(PRM).read()

    @pytest.fixture()
    def outfile(self, tmpdir: str) -> Path:
        return Path(tmpdir) / "out.prm"

    def test_parameters(self, u: Mapping[str, pd.DataFrame], outfile: Path):
        with mda.Writer(outfile, nonbonded=True) as ofile:
            ofile.write(u)

        u2 = ParamReader(outfile).read()
        assert_allclose(u["ATOMS"]["mass"], u2["ATOMS"]["mass"],
                        err_msg="The atomic masses don't match.")
        assert_allclose(u["BONDS"]["Kb"], u2["BONDS"]["Kb"],
                        err_msg="The force constants don't match.")

    def test_roundtrip(self, u: Mapping[str, pd.DataFrame], outfile: Path):
        # Write out a copy of the internal coordinates, and compare this against
        # the original. This is more rigorous as it checks all formatting.
        with mda.Writer(outfile, nonbonded=True) as ofile:
            ofile.write(u)

        def PRM_iter(fn: Union[str, Path]):
            with open(fn) as inf:
                for line in inf:
                    if not line.startswith('*'):
                        yield line

        for ref, other in zip(PRM_iter(PRM), PRM_iter(outfile)):
            assert ref == other
