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
"""Tests for solvent ion model."""

from typing import ClassVar, List, Mapping, Tuple

import numpy as np
from MDAnalysis.core.topologyattrs import Atomtypes, Bonds

from .base import ModelBase


class Water(ModelBase):
    """Create a universe consisting of the water oxygen."""
    model: ClassVar[str] = "WATER"
    describe: ClassVar[str] = "c.o.m./c.o.g. of whole water molecule"

    def __init__(self, xplor: bool = True, extended: bool = True,
                 com: bool = True, guess_angles: bool = True,
                 cutoff: float = 10.0):
        super().__init__(xplor, extended, com, guess_angles, cutoff)

        self._guess: bool = False
        self._mapping["OW"]: str = "water"
        self._selection.update(self._mapping)

    def _add_atomtypes(self):
        n_atoms: int = self.universe.atoms.n_atoms
        atomtypes: Atomtypes = Atomtypes(np.ones(n_atoms))
        self.universe.add_TopologyAttr(atomtypes)

    def _add_bonds(self):
        self.universe.add_TopologyAttr(Bonds([]))


class Tip3p(ModelBase):
    """Create a universe containing all three water atoms."""
    model: ClassVar[str] = "TIP3P"
    describe: ClassVar[str] = "All-atom watter"

    def __init__(self, xplor: bool = True, extended: bool = True,
                 com: bool = True, guess_angles: bool = True,
                 cutoff: float = 10.0):
        super().__init__(xplor, extended, com, guess_angles, cutoff)

        self._mapping["OW"]: str = "name OW MW"
        self._mapping["HW1"]: str = "name HW1"
        self._mapping["HW2"]: str = "name HW2"
        self._selection.update(self._mapping)
        self._types: Mapping[str, int] = {
            key: value + 1
            for key, value in zip(self._mapping.keys(),
                                  range(len(self._mapping)))
        }

    def _add_atomtypes(self):
        atomtypes: List[int] = [
            self._types[atom.name] for atom in self.universe.atoms
        ]
        self.universe.add_TopologyAttr(Atomtypes(atomtypes))

    def _add_bonds(self):
        bonds: List[Tuple[int, int]] = []
        bonds.extend([
            _
            for s in self.universe.segments
            for _ in zip(s.atoms.select_atoms("name OW").ix,
                         s.atoms.select_atoms("name HW1").ix)
        ])
        bonds.extend([
            _
            for s in self.universe.segments
            for _ in zip(s.atoms.select_atoms("name OW").ix,
                         s.atoms.select_atoms("name HW2").ix)
        ])
        bonds.extend([
            _
            for s in self.universe.segments
            for _ in zip(s.atoms.select_atoms("name HW1").ix,
                         s.atoms.select_atoms("name HW2").ix)
        ])
        self.universe.add_TopologyAttr(Bonds(bonds))


class Dma(ModelBase):
    """Create a universe for N-dimethylacetamide."""
    model: ClassVar[str] = "DMA"
    describe: ClassVar[str] = "c.o.m./c.o.g. of C1, N, C2, and C3 of DMA"

    def __init__(self, xplor: bool = True, extended: bool = True,
                 com: bool = True, guess_angles: bool = True,
                 cutoff: float = 10.0):
        super().__init__(xplor, extended, com, guess_angles, cutoff)

        self._mapping["C1"]: str = "resname DMA and name C1 H1*"
        self._mapping["N"]: str = "resname DMA and name C N O"
        self._mapping["C2"]: str = "resname DMA and name C2 H2*"
        self._mapping["C3"]: str = "resname DMA and name C3 H3*"
        self._selection.update(self._mapping)
        self._types: Mapping[str, int] = {
            key: value + 4
            for key, value in zip(self._mapping.keys(),
                                  range(len(self._mapping)))
        }

    def _add_atomtypes(self):
        atomtypes: List[int] = [
            self._types[atom.name] for atom in self.universe.atoms
        ]
        self.universe.add_TopologyAttr(Atomtypes(atomtypes))

    def _add_bonds(self):
        bonds: List[Tuple[int, int]] = []
        bonds.extend([
            idx
            for segment in self.universe.segments
            for idx in zip(segment.atoms.select_atoms("name C1").ix,
                           segment.atoms.select_atoms("name N").ix)
        ])
        bonds.extend([
            idx
            for segment in self.universe.segments
            for idx in zip(segment.atoms.select_atoms("name C2").ix,
                           segment.atoms.select_atoms("name N").ix)
        ])
        bonds.extend([
            idx
            for segment in self.universe.segments
            for idx in zip(segment.atoms.select_atoms("name C3").ix,
                           segment.atoms.select_atoms("name N").ix)
        ])
        self.universe.add_TopologyAttr(Bonds(bonds))
