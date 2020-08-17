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
#   Timothy H. Click, Nixon Raj, and Jhih-Wei Chuniverse.
#   Simulation. Meth Enzymology. 578 (2016), 327-342,
#   Calculation of Enzyme Fluctuograms from All-Atom Molecular Dynamics
#   doi:10.1016/bs.mie.2016.05.024.
#
# ------------------------------------------------------------------------------

from collections import OrderedDict
from pathlib import Path
from typing import List
from unittest.mock import patch

import MDAnalysis as mda
import pytest
from MDAnalysisTests import make_Universe
from numpy.testing import assert_equal

from ..datafiles import COR


class TestCORWriter:
    @pytest.fixture()
    def universe(self) -> mda.Universe:
        return mda.Universe(COR)

    def test_writer(self, universe: mda.Universe, tmp_path: Path):
        filename: Path = tmp_path / "temp.cor"
        with patch("fluctmatch.parsers.writers.COR.Writer.write") as writer, mda.Writer(
            filename, n_atoms=universe.atoms.n_atoms
        ) as w:
            w.write(universe.atoms)
            writer.assert_called()

    def test_roundtrip(self, universe: mda.Universe, tmp_path: Path):
        # Write out a copy of the Universe, and compare this against the
        # original. This is more rigorous than simply checking the coordinates
        # as it checks all formatting
        filename: Path = tmp_path / "temp.cor"
        with mda.Writer(filename, n_atoms=universe.atoms.n_atoms) as w:
            w.write(universe.atoms)

        def CRD_iter(fn):
            with open(fn, "r") as inf:
                for line in inf:
                    if not line.startswith("*"):
                        yield line

        for ref, other in zip(CRD_iter(COR), CRD_iter(filename)):
            assert ref == other

    def test_write_atoms(self, universe: mda.Universe, tmp_path: Path):
        # Test that written file when read gives same coordinates
        filename: Path = tmp_path / "temp.cor"
        with mda.Writer(filename, n_atoms=universe.atoms.n_atoms) as w:
            w.write(universe.atoms)

        u2 = mda.Universe(filename)

        assert_equal(universe.atoms.positions, u2.atoms.positions)


class TestCORWriterMissingAttrs:
    # All required attributes with the default value
    req_attrs = OrderedDict(
        [("resnames", "UNK"), ("resids", 1), ("names", "X"), ("tempfactors", 0.0)]
    )

    @pytest.mark.parametrize("missing_attr", req_attrs)
    def test_warns(self, missing_attr: OrderedDict, tmp_path: Path):
        attrs: List[str, ...] = list(self.req_attrs.keys())
        attrs.remove(missing_attr)
        universe: mda.Universe = make_Universe(attrs, trajectory=True)

        outfile: Path = tmp_path / "out.cor"
        with pytest.warns(UserWarning), mda.Writer(
            outfile, n_atoms=universe.atoms.n_atoms
        ) as w:
            w.write(universe.atoms)

    @pytest.mark.parametrize("missing_attr", req_attrs)
    def test_write(self, missing_attr: OrderedDict, tmp_path: Path):
        attrs: List[str, ...] = list(self.req_attrs.keys())
        attrs.remove(missing_attr)
        universe: mda.Universe = make_Universe(attrs, trajectory=True)

        outfile: Path = tmp_path / "out.cor"
        with pytest.warns(UserWarning), mda.Writer(
            outfile, n_atoms=universe.atoms.n_atoms
        ) as w:
            w.write(universe.atoms)

        u2 = mda.Universe(outfile)

        # Check all other attrs aren't disturbed
        for attr in attrs:
            assert_equal(getattr(universe.atoms, attr), getattr(u2.atoms, attr))
        # Check missing attr is as expected
        assert_equal(getattr(u2.atoms, missing_attr), self.req_attrs[missing_attr])
