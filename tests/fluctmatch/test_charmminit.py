# fluctmatch --- https://github.com/tclick/python-fluctmatch
# Copyright (c) 2013-2020 The fluctmatch Development Team and contributors
# (see the file AUTHORS for the full list of names)
#
# Released under the New BSD license.
#
# Please cite your use of fluctmatch in published work:
#
# Timothy H. Click, Nixon Raj, and Jhih-Wei Chu.
# Calculation of Enzyme Fluctuograms from All-Atom Molecular Dynamics
# Simulation. Meth Enzymology. 578 (2016), 327-342,
# doi:10.1016/bs.mie.2016.05.024.
#

import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
import static_frame as sf

from fluctmatch.fluctmatch.plugins.fluctmatch import FluctMatch
from tests.datafiles import IC


class TestCharmmInit:
    def test_simulate(self):
        with tempfile.NamedTemporaryFile(mode="r+", suffix="inp") as infile:
            directory = Path(infile.name).parent
            simulator = FluctMatch(output_dir=directory)
            with patch("subprocess.check_call") as check_call:
                simulator.simulate(input_dir=directory)
                check_call.assert_called()

    def test_simulate_error(self):
        with tempfile.NamedTemporaryFile(mode="r+", suffix="inp") as infile:
            directory = Path(infile.name).parent
            simulator = FluctMatch(output_dir=directory)
            pytest.raises(
                IOError, simulator.simulate, input_dir=directory, executable=None
            )

    def test_calculate(self):
        with tempfile.NamedTemporaryFile(mode="r+", suffix="inp") as infile:
            directory = Path(infile.name).parent
            simulator = FluctMatch(output_dir=directory)
            simulator.data["average"] = IC
            simulator.data["fluctuation"] = simulator.data["average"]
            parameters: sf.Frame = simulator.calculate()
            assert isinstance(parameters, sf.Frame)
            assert parameters.size > 0

    def test_simulate_calculate(self):
        with tempfile.NamedTemporaryFile(mode="r+", suffix="inp") as infile:
            directory = Path(infile.name).parent
            simulator = FluctMatch(output_dir=directory)
            with patch("subprocess.check_call") as check_call:
                simulator.simulate_calculate(input_dir=directory)
                check_call.assert_called()
