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
"""Class definition for beads using C-alpha and C-beta positions"""

from typing import ClassVar
from typing import List
from typing import Mapping
from typing import NoReturn
from typing import Tuple

import MDAnalysis as mda
from MDAnalysis.core.topologyattrs import Bonds

from ..base import ModelBase
from ..selection import *


class Model(ModelBase):
    """Universe consisting of the C-alpha and sidechains of a protein."""

    description: ClassVar[
        str
    ] = "C-alpha and sidechain (c.o.m./c.o.g.) of protein"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self._mapping: Mapping[str, str] = dict(
            CA="calpha", CB="hsidechain and not name H*", ions="bioion"
        )
        self._selection: Mapping[str, str] = dict(
            CA="hbackbone", CB="hsidechain", ions="bioion"
        )

    def _add_bonds(self) -> NoReturn:
        bonds: List[Tuple[int, int]] = []

        # Create bonds intraresidue C-alpha and C-beta atoms.
        residues = self.universe.select_atoms(
            "protein and not resname GLY"
        ).residues
        atom1: mda.AtomGroup = residues.atoms.select_atoms("calpha")
        atom2: mda.AtomGroup = residues.atoms.select_atoms("cbeta")
        bonds.extend(list(zip(atom1.ix, atom2.ix)))

        # Create interresidue C-alpha bonds within a segment
        for segment in self.universe.segments:
            atoms: mda.AtomGroup = segment.atoms.select_atoms("calpha")
            bonds.extend(list(zip(atoms.ix[1:], atoms.ix[:-1])))

        self.universe.add_TopologyAttr(Bonds(bonds))
