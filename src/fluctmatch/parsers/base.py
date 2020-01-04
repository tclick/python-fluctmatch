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

import os
import time
from typing import Mapping
from typing import Union

import MDAnalysis as mda
from MDAnalysis.coordinates.base import IOBase
from MDAnalysis.coordinates.base import _Readermeta
from MDAnalysis.coordinates.base import _Writermeta


class TopologyReaderBase(IOBase, metaclass=_Readermeta):
    def __init_subclass__(cls, **kwargs: Mapping):
        super().__init_subclass__(**kwargs)

    def read(self):  # pragma: no cover
        """Read the file"""
        raise NotImplementedError("Override this in each subclass")


class TopologyWriterBase(IOBase, metaclass=_Writermeta):
    def __init_subclass__(cls, **kwargs: Mapping):
        super().__init_subclass__(**kwargs)

    def __init__(self):
        self.title: str = f"""
            * Created by fluctmatch on {time.asctime(time.localtime())}
            * User: {os.environ["USER"]}
            """.strip("\n")

    def write(self, selection: Union[mda.Universe, mda.AtomGroup]):
        """Write selection at current trajectory frame to file.

        Parameters
        ----------
        selection : AtomGroup
             group of atoms to be written

        """
        raise NotImplementedError("Override this in each subclass")
