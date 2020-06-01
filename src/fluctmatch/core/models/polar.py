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
"""Class definition for beads using N, carboxyl oxygens, and polar sidechains.
"""

from typing import ClassVar, Mapping

from ..selection import *
from . import ncsc


class Model(ncsc.Model):
    """Universe consisting of the amine, carboxyl, and polar regions."""

    model: ClassVar[str] = "POLAR"
    description: ClassVar[
        str
    ] = "c.o.m./c.o.g. of N, C, and polar sidechains of protein"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self._mapping["CB"]: Mapping[str, str] = dict(
            ALA="name CB",
            ARG="name NH*",
            ASN="name OD1 ND2",
            ASP="name OD*",
            CYS="name SG",
            GLN="name OE1 NE2",
            GLU="name OE*",
            HIS="name CG ND1 CD2 CE1 NE2",
            HSD="name CG ND1 CD2 CE1 NE2",
            HSE="name CG ND1 CD2 CE1 NE2",
            HSP="name CG ND1 CD2 CE1 NE2",
            ILE="name CG1 CG2 CD",
            LEU="name CD1 CD2",
            LYS="name NZ",
            MET="name SD",
            PHE="name CG CD* CE* CZ",
            PRO="name CG",
            SER="name OG",
            THR="name OG1",
            TRP="name CG CD* NE CE* CZ* CH",
            TYR="name CG CD* CE* CZ OH",
            VAL="name CG1 CG2",
        )
