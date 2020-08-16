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
"""Tests for TIP3P water model."""

from collections import namedtuple
from itertools import combinations
from typing import List, Mapping, Tuple

from MDAnalysis.core.topologyattrs import Bonds

from ..base import ModelBase


class Model(ModelBase):
    """Create a universe containing all three water atoms."""

    model = "TIP3P"
    description = "All-atom watter"

    def __init__(
        self,
        *,
        xplor: bool = True,
        extended: bool = True,
        com: bool = True,
        guess_angles: bool = False,
        rmin: float = 0.0,
        rmax: float = 10.0,
    ) -> None:
        super().__init__(
            xplor=xplor,
            extended=extended,
            com=com,
            guess_angles=guess_angles,
            rmin=rmin,
            rmax=rmax,
        )

        BEADS = namedtuple("BEADS", "OW HW1 HW2")
        self._mapping = BEADS(OW="name OW", HW1="name HW1", HW2="name HW2")
        self._selection = BEADS(OW="name OW MW", HW1="name HW1", HW2="name HW2")
        self._types: Mapping[str, int] = {
            key: value + 1
            for key, value in zip(self._mapping._fields, range(len(self._mapping)))
        }

    def _add_bonds(self) -> None:
        bonds: List[Tuple[int, int]] = []
        for segment in self._universe.segments:
            for select1, select2 in combinations(self._selection._fields, 2):
                atom1 = getattr(self._selection, select1)
                atom2 = getattr(self._selection, select2)
                bonds.extend(
                    tuple(
                        zip(
                            segment.atoms.select_atoms(atom1).ix,
                            segment.atoms.select_atoms(atom2).ix,
                        )
                    )
                )

        self._universe.add_TopologyAttr(Bonds(bonds))
