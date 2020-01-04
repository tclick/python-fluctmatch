# -*- coding: utf-8 -*-
#
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
#  Timothy H. Click, Nixon Raj, and Jhih-Wei Chu.
#  Calculation of Enzyme Fluctuograms from All-Atom Molecular Dynamics
#  Simulation. Meth Enzymology. 578 (2016), 327-342,
#  doi:10.1016/bs.mie.2016.05.024.
"""Tests for TIP3P water model."""

from typing import ClassVar
from typing import List
from typing import Mapping
from typing import NoReturn
from typing import Tuple

import MDAnalysis as mda
from MDAnalysis.core.topologyattrs import Atomtypes
from MDAnalysis.core.topologyattrs import Bonds

from ..base import ModelBase


class Model(ModelBase):
    """Create a universe containing all three water atoms."""

    description: ClassVar[str] = "All-atom watter"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self._mapping["OW"]: str = "name OW MW"
        self._mapping["HW1"]: str = "name HW1"
        self._mapping["HW2"]: str = "name HW2"
        self._selection.update(self._mapping)
        self._types: Mapping[str, int] = {
            key: value + 1
            for key, value in
            zip(self._mapping.keys(), range(len(self._mapping)))
        }

    def _add_atomtypes(self) -> NoReturn:
        atomtypes: List[int] = [self._types[atom.name] for atom in
                                self.universe.atoms]
        self.universe.add_TopologyAttr(Atomtypes(atomtypes))

    def _add_bonds(self) -> NoReturn:
        bonds: List[Tuple[int, int]] = []
        for segment in self.universe.segments:
            atom1: mda.AtomGroup = segment.atoms.select_atoms("name OW")
            atom2: mda.AtomGroup = segment.atoms.select_atoms("name HW1")
            atom3: mda.AtomGroup = segment.atoms.select_atoms("name HW2")
            bonds.extend(list(zip(atom1.ix, atom2.ix)))
            bonds.extend(list(zip(atom1.ix, atom3.ix)))
            bonds.extend(list(zip(atom2.ix, atom3.ix)))

        self.universe.add_TopologyAttr(Bonds(bonds))
