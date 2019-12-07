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

from collections import OrderedDict
from pathlib import Path

import MDAnalysis as mda
import pytest
from MDAnalysisTests import make_Universe
from numpy.testing import assert_equal

import fluctmatch
from ..datafiles import COR


class TestCORWriter(object):
    @pytest.fixture()
    def u(self):
        print(mda._READERS)
        return mda.Universe(COR)

    @pytest.fixture()
    def outfile(self, tmpdir) -> Path:
        return Path(tmpdir) / 'out.cor'

    def test_roundtrip(self, u, outfile):
        # Write out a copy of the Universe, and compare this against the
        # original. This is more rigorous than simply checking the coordinates
        # as it checks all formatting
        u.atoms.write(outfile)

        def CRD_iter(fn):
            with open(fn, 'r') as inf:
                for line in inf:
                    if not line.startswith('*'):
                        yield line

        for ref, other in zip(CRD_iter(COR), CRD_iter(outfile)):
            assert ref == other

    def test_write_atoms(self, u, outfile):
        # Test that written file when read gives same coordinates
        u.atoms.write(outfile)

        u2 = mda.Universe(outfile)

        assert_equal(u.atoms.positions, u2.atoms.positions)


class TestCORWriterMissingAttrs(object):
    # All required attributes with the default value
    req_attrs = OrderedDict([
        ('resnames', 'UNK'),
        ('resids', 1),
        ('names', 'X'),
        ('tempfactors', 0.0),
    ])

    @pytest.mark.parametrize('missing_attr', req_attrs)
    def test_warns(self, missing_attr, tmpdir):
        attrs = list(self.req_attrs.keys())
        attrs.remove(missing_attr)
        u = make_Universe(attrs, trajectory=True)

        outfile = str(tmpdir) + '/out.cor'
        with pytest.warns(UserWarning):
            u.atoms.write(outfile)

    @pytest.mark.parametrize('missing_attr', req_attrs)
    def test_write(self, missing_attr, tmpdir):
        attrs = list(self.req_attrs.keys())
        attrs.remove(missing_attr)
        u = make_Universe(attrs, trajectory=True)

        outfile = str(tmpdir) + '/out.cor'
        u.atoms.write(outfile)
        u2 = mda.Universe(outfile)

        # Check all other attrs aren't disturbed
        for attr in attrs:
            assert_equal(getattr(u.atoms, attr),
                         getattr(u2.atoms, attr))
        # Check missing attr is as expected
        assert_equal(getattr(u2.atoms, missing_attr),
                     self.req_attrs[missing_attr])
