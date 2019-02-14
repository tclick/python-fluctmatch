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
#  Simulation. Meth Enzymology. 578 (2016), 327-342,
#  Calculation of Enzyme Fluctuograms from All-Atom Molecular Dynamics
#  doi:10.1016/bs.mie.2016.05.024.
"""Classes for various nucleic acid models."""

from typing import ClassVar, List, Tuple, Mapping

import numpy as np
import MDAnalysis as mda
from MDAnalysis.core.topologyattrs import Bonds, Charges

from .base import ModelBase


class Nucleic3(ModelBase):
    """A universe the phosphate, sugar, and base of the nucleic acid."""
    model: ClassVar[str] = "NUCLEIC3"
    describe: ClassVar[str] = "Phosohate, sugar, and nucleotide of nucleic acid"

    def __init__(self, xplor: bool = True, extended: bool = True,
                 com: bool = True, guess_angles: bool = True,
                 cutoff: float = 10.0):
        super().__init__(xplor, extended, com, guess_angles, cutoff)

        self._mapping["P"]: str = "nucleicphosphate and not name H*"
        self._mapping["C4'"]: str = "hnucleicsugar and not name H*"
        self._mapping["C5"]: str = "hnucleicbase and not name H*"
        self._selection: Mapping[str, str] = {
            "P": "nucleicphosphate",
            "C4'": "hnucleicsugar",
            "C5": "hnucleicbase"
        }

    def _add_bonds(self):
        bonds: List[Tuple[int, int]] = []
        bonds.extend([
            idx 
            for segment in self.universe.segments 
            for idx in zip(segment.atoms.select_atoms("name P").ix,
                           segment.atoms.select_atoms("name C4'").ix)
        ])
        bonds.extend([
            idx 
            for segment in self.universe.segments 
            for idx in zip(segment.atoms.select_atoms("name C4'").ix,
                           segment.atoms.select_atoms("name C5").ix)
        ])
        bonds.extend([
            idx 
            for segment in self.universe.segments 
            for idx in zip(segment.atoms.select_atoms("name C4'").ix[:-1],
                           segment.atoms.select_atoms("name P").ix[1:])
            if segment.residues.n_residues > 1
        ])
        self.universe.add_TopologyAttr(Bonds(bonds))


class Nucleic4(ModelBase):
    """A universe of the phosphate, C4', C3', and base of the nucleic acid."""
    model: ClassVar[str] = "NUCLEIC4"
    describe: ClassVar[str] = ("Phosphate, C2', C4', and c.o.m./c.o.g. of C4/C5 of "
                     "nucleic acid")

    def __init__(self, xplor: bool = True, extended: bool = True,
                 com: bool = True, guess_angles: bool = True,
                 cutoff: float = 10.0):
        super().__init__(xplor, extended, com, guess_angles, cutoff)

        self._mapping["P"]: str = "nucleicphosphate and not name H*"
        self._mapping["C4'"]: str = "name C4'"
        self._mapping["C2'"]: str = "name C2'"
        self._mapping["C5"]: str = "nucleiccenter and not name H*"
        self._selection: Mapping[str, str] = {
            "P": "nucleicphosphate",
            "C4'": "sugarC4",
            "C2'": "sugarC2",
            "C5": "hnucleicbase"
        }

    def _add_bonds(self):
        bonds: List[Tuple[int, int]] = []
        bonds.extend([
            idx
            for segment in self.universe.segments
            for idx in zip(segment.atoms.select_atoms("name P").ix,
                           segment.atoms.select_atoms("name C4'").ix)
        ])
        bonds.extend([
            idx
            for segment in self.universe.segments
            for idx in zip(segment.atoms.select_atoms("name C4'").ix,
                           segment.atoms.select_atoms("name C2'").ix)
        ])
        bonds.extend([
            idx
            for segment in self.universe.segments
            for idx in zip(segment.atoms.select_atoms("name C4'").ix,
                           segment.atoms.select_atoms("name C5").ix)
        ])
        bonds.extend([
            idx
            for segment in self.universe.segments
            for idx in zip(segment.atoms.select_atoms("name C4'").ix[:-1],
                           segment.atoms.select_atoms("name P").ix[1:])
        ])
        self.universe.add_TopologyAttr(Bonds(bonds))


class Nucleic6(ModelBase):
    """A universe accounting for six sites involved with hydrogen bonding."""
    model: ClassVar[str] = "NUCLEIC6"
    describe: ClassVar[str] = "Phosphate, C2', C4', and 3 sites on the nucleotide"

    def __init__(self, xplor: bool = True, extended: bool = True,
                 com: bool = True, guess_angles: bool = True,
                 cutoff: float = 10.0):
        super().__init__(xplor, extended, com, guess_angles, cutoff)

        self._mapping["P"]: str = "name P H5T"
        self._mapping["C4'"]: str = "name C4'"
        self._mapping["C2'"]: str = "name C2'"
        self._mapping["H1"]: str = ("(resname ADE DA* RA* and name N6) or "
                                    "(resname OXG GUA DG* RG* and name O6) or "
                                    "(resname CYT DC* RC* and name N4) or "
                                    "(resname THY URA DT* RU* and name O4)")
        self._mapping["H2"]: str = (
            "(resname ADE DA* RA* OXG GUA DG* RG* and name N1) or "
            "(resname CYT DC* RC* THY URA DT* RU* and name N3)")
        self._mapping["H3"]: str = (
            "(resname ADE DA* RA* and name H2) or "
            "(resname OXG GUA DG* RG* and name N2) or "
            "(resname CYT DC* RC* THY URA DT* RU* and name O2)")
        self._selection: Mapping[str, str] = {
            "P": "nucleicphosphate",
            "C4'": "sugarC4",
            "C2'": "sugarC2",
            "H1": "nucleic and name C2'",
            "H2": "nucleic and name C2'",
            "H3": "nucleic and name C2'"
        }

    def create_topology(self, universe: mda.Universe):
        """Deteremine the topology attributes and initialize the universe.

        Parameters
        ----------
        universe : :class:`~MDAnalysis.Universe`
            An all-atom universe
        """
        super().create_topology(universe)

        charges: np.ndarray = np.zeros(self.universe.atoms.n_atoms,
                                       dtype=np.float32)
        self.universe.add_TopologyAttr(Charges(charges))

    def _add_bonds(self):
        bonds: List[Tuple[int, int]] = []
        bonds.extend([
            idx
            for segment in self.universe.segments
            for idx in zip(segment.atoms.select_atoms("name P").ix,
                           segment.atoms.select_atoms("name C4'").ix)
        ])
        bonds.extend([
            idx
            for segment in self.universe.segments
            for idx in zip(segment.atoms.select_atoms("name C4'").ix,
                           segment.atoms.select_atoms("name C2'").ix)
        ])
        bonds.extend([
            idx
            for segment in self.universe.segments
            for idx in zip(segment.atoms.select_atoms("name C2'").ix,
                           segment.atoms.select_atoms("name H1").ix)
        ])
        bonds.extend([
            idx
            for segment in self.universe.segments
            for idx in zip(segment.atoms.select_atoms("name H1").ix,
                           segment.atoms.select_atoms("name H2").ix)
        ])
        bonds.extend([
            idx
            for segment in self.universe.segments
            for idx in zip(segment.atoms.select_atoms("name H2").ix,
                           segment.atoms.select_atoms("name H3").ix)
        ])
        bonds.extend([
            idx
            for segment in self.universe.segments
            for idx in zip(segment.atoms.select_atoms("name C4'").ix[:-1],
                           segment.atoms.select_atoms("name P").ix[1:])
        ])
        self.universe.add_TopologyAttr(Bonds(bonds))
