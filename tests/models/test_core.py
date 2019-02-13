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

import MDAnalysis as mda
import pytest
from numpy import testing

from fluctmatch.models import core, protein
from ..datafiles import TPR, XTC


class TestModeller:
    @pytest.fixture(scope="class")
    def u(self) -> mda.Universe:
        return mda.Universe(TPR, XTC)

    @pytest.fixture(scope="class")
    def u2(self) -> mda.Universe:
        return core.modeller(TPR, XTC)

    @pytest.fixture(scope="class")
    def system(self) -> protein.Polar:
        return protein.Polar()

    def test_creation(self, u: mda.Universe, u2: mda.Universe,
                      system: protein.Polar):
        u3 = system.transform(u)

        testing.assert_raises(AssertionError, testing.assert_equal,
                              (u.atoms.n_atoms,), (u2.atoms.n_atoms,))
        testing.assert_equal(u2.atoms.names, u3.atoms.names,
                             err_msg="Universes don't match.")
