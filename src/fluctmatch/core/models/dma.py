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
"""Tests for DMA solvent model."""

from collections import namedtuple
from typing import List, Mapping, Tuple

from MDAnalysis.core.groups import AtomGroup
from MDAnalysis.core.topologyattrs import Bonds

from ..base import ModelBase


class Model(ModelBase):
    """Create a universe for N-dimethylacetamide."""

    model = "DNA"
    description = "c.o.m./c.o.g. of C1, N, C2, and C3 of DMA"

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

        BEADS = namedtuple("BEADS", "C1 N C2 C3")
        self._mapping = BEADS(
            C1="resname DMA and name C1 H1*",
            N="resname DMA and name C N O",
            C2="resname DMA and name C2 H2*",
            C3="resname DMA and name C3 H3*",
        )
        self._selection = self._mapping

        self._types: Mapping[str, int] = {
            key: value + 4
            for key, value in zip(self._mapping._fields, range(len(self._mapping)))
        }

    def _add_bonds(self) -> None:
        bonds: List[Tuple[int, int]] = []
        for segment in self._universe.segments:
            atom1: AtomGroup = segment.atoms.select_atoms("name C1")
            atom2: AtomGroup = segment.atoms.select_atoms("name N")
            atom3: AtomGroup = segment.atoms.select_atoms("name C2")
            atom4: AtomGroup = segment.atoms.select_atoms("name C3")
            bonds.extend(tuple(zip(atom1.ix, atom2.ix)))
            bonds.extend(tuple(zip(atom2.ix, atom3.ix)))
            bonds.extend(tuple(zip(atom2.ix, atom4.ix)))

        self._universe.add_TopologyAttr(Bonds(bonds))
