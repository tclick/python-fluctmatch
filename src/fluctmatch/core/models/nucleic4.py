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
"""Class for 4-bead nucleic acid."""

from typing import ClassVar, List, Mapping, NoReturn, Tuple

from MDAnalysis.core.topologyattrs import Bonds

from ..base import ModelBase
from ..selection import *


class Model(ModelBase):
    """A universe of the phosphate, C4', C3', and base of the nucleic acid."""

    description: ClassVar[str] = (
        "Phosphate, C2', C4', and c.o.m./c.o.g. of C4/C5 of nucleic acid"
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self._mapping["P"]: str = "nucleicphosphate and not name H*"
        self._mapping["C4'"]: str = "name C4'"
        self._mapping["C2'"]: str = "name C2'"
        self._mapping["C5"]: str = "nucleiccenter and not name H*"
        self._selection: Mapping[str, str] = {
            "P": "nucleicphosphate",
            "C4'": "sugarC4",
            "C2'": "sugarC2",
            "C5": "hnucleicbase",
        }

    def _add_bonds(self) -> NoReturn:
        bonds: List[Tuple[int, int]] = []
        for segment in self.universe.segments:
            atom1 = segment.atoms.select_atoms("name P")
            atom2 = segment.atoms.select_atoms("name C4'")
            atom3 = segment.atoms.select_atoms("name C2'")
            atom4 = segment.atoms.select_atoms("name C5")

            bonds.extend(list(zip(atom1.ix, atom2.ix)))
            bonds.extend(list(zip(atom2.ix, atom3.ix)))
            bonds.extend(list(zip(atom2.ix, atom4.ix)))
            bonds.extend(list(zip(atom2.ix[:-1], atom1.ix[1:])))

        self.universe.add_TopologyAttr(Bonds(bonds))
