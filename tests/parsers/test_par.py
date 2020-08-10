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
from typing import Mapping, NamedTuple, Union
from unittest.mock import patch

import MDAnalysis as mda
import pytest
import static_frame as sf
from numpy.testing import assert_allclose

import fluctmatch.parsers.readers.PAR as ParamReader
from ..datafiles import PAR


class TestPARReader:
    def test_reader(self):
        with ParamReader.Reader(PAR) as infile:
            parameters = infile.read()

        assert isinstance(parameters, ParamReader.Reader._HEADERS)
        assert parameters.ATOMS.size > 0
        assert parameters.BONDS.size > 0
        assert parameters.ANGLES.size == 0


class TestPARWriter:
    @pytest.fixture()
    def parameters(self) -> NamedTuple:
        return ParamReader.Reader(PAR).read()

    def test_writer(self, parameters: sf.Frame, tmp_path: Path):
        filename = tmp_path / "temp.par"
        with patch("fluctmatch.parsers.writers.PAR.Writer.write") as writer, mda.Writer(
            filename, nonbonded=True
        ) as ofile:
            ofile.write(parameters)
            writer.assert_called()

    def test_parameters(self, parameters: Mapping[str, sf.Frame], tmp_path: Path):
        filename = tmp_path / "temp.par"
        with mda.Writer(filename, nonbonded=True) as ofile:
            ofile.write(parameters)

        new_parameters: NamedTuple = ParamReader.Reader(filename).read()
        assert_allclose(
            parameters.ATOMS["mass"],
            new_parameters.ATOMS["mass"],
            err_msg="The atomic masses don't match.",
        )
        assert_allclose(
            parameters.BONDS["Kb"],
            new_parameters.BONDS["Kb"],
            err_msg="The force constants don't match.",
        )

    def test_roundtrip(self, parameters: Mapping[str, sf.Frame], tmp_path: Path):
        # Write out a copy of the internal coordinates, and compare this against
        # the original. This is more rigorous as it checks all formatting.
        filename = tmp_path / "temp.par"
        with mda.Writer(filename, nonbonded=True) as ofile:
            ofile.write(parameters)

        def PAR_iter(fn: Union[str, Path]):
            with open(fn) as input_file:
                for line in input_file:
                    if not line.startswith("*"):
                        yield line

        for ref, other in zip(PAR_iter(PAR), PAR_iter(filename)):
            assert ref == other
