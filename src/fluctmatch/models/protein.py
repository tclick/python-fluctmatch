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

from typing import List, Tuple, Mapping

from MDAnalysis.core.topologyattrs import Bonds

from .base import ModelBase
from .selection import *


class Calpha(ModelBase):
    """Create a universe defined by the protein C-alpha."""
    model: str = "CALPHA"
    describe: str = "C-alpha of a protein"

    def __init__(self,
                 xplor: bool = True,
                 extended: bool = True,
                 com: bool = True,
                 guess_angles: bool = True,
                 cutoff: float = 10.0):
        super().__init__(xplor, extended, com, guess_angles, cutoff)

        self._mapping["CA"]: str = "calpha"
        self._mapping["ions"]: str = "bioion"
        self._selection: Mapping[str, str] = dict(CA="protein", ions="bioion")

    def _add_bonds(self):
        bonds: List[Iterable[int, int]] = []
        bonds.extend([
            _
            for s in self.universe.segments for _ in zip(
                s.atoms.select_atoms("calpha").ix,
                s.atoms.select_atoms("calpha").ix[1:])
        ])
        self.universe._topology.add_TopologyAttr(Bonds(bonds))
        self.universe._generate_from_topology()


class Caside(ModelBase):
    """Create a universe consisting of the C-alpha and sidechains of a protein.
    """
    model: str = "CASIDE"
    describe: str = "C-alpha and sidechain (c.o.m./c.o.g.) of protein"

    def __init__(self,
                 xplor: bool = True,
                 extended: bool = True,
                 com: bool = True,
                 guess_angles: bool = True,
                 cutoff: float = 10.0):
        super().__init__(xplor, extended, com, guess_angles, cutoff)

        self._mapping["CA"]: str = "calpha"
        self._mapping["CB"]: str = "hsidechain and not name H*"
        self._mapping["ions"]: str = "bioion"
        self._selection: Mapping[str, str] = dict(CA="hbackbone",
                                                  CB="hsidechain",
                                                  ions="bioion")

    def _add_bonds(self):
        bonds: List[Tuple[int, int]] = []
        bonds.extend([
            _
            for s in self.universe.segments
            for _ in zip(s.atoms.select_atoms("calpha").ix,
                         s.atoms.select_atoms("calpha").ix[1:])
        ])
        bonds.extend([(
            r.atoms.select_atoms("calpha").ix[0],
            r.atoms.select_atoms("cbeta").ix[0])
            for r in self.universe.residues
            if (r.resname != "GLY"
                and r.resname in selection.ProteinSelection.prot_res)
        ])
        self.universe._topology.add_TopologyAttr(Bonds(bonds))
        self.universe._generate_from_topology()


class Ncsc(ModelBase):
    """Create a universe consisting of the amine, carboxyl, and sidechain regions.
    """
    model: str = "NCSC"
    describe: str = "c.o.m./c.o.g. of N, O, and sidechain of protein"

    def __init__(self,
                 xplor: bool = True,
                 extended: bool = True,
                 com: bool = True,
                 guess_angles: bool = True,
                 cutoff: float = 10.0):
        super().__init__(xplor, extended, com, guess_angles, cutoff)

        self._mapping["N"]: str = "protein and name N"
        self._mapping["CB"]: str = "hsidechain and not name H*"
        self._mapping["O"]: str = "protein and name O OT1 OT2 OXT"
        self._mapping["ions"]: str = "bioion"
        self._selection: Mapping[str, str] = dict(N="amine",
                                                  CB="hsidechain",
                                                  O="carboxyl",
                                                  ions="bioion")

    def _add_bonds(self):
        bonds: List[Tuple[int, int]] = []
        bonds.extend([
            _
            for s in self.universe.segments
            for _ in zip(s.atoms.select_atoms("name N").ix,
                         s.atoms.select_atoms("name O").ix)
        ])
        bonds.extend([(
            r.atoms.select_atoms("name N").ix[0],
            r.atoms.select_atoms("cbeta").ix[0])
            for r in self.universe.residues
            if (r.resname != "GLY"
                and r.resname in selection.ProteinSelection.prot_res)
        ])
        bonds.extend([(
            r.atoms.select_atoms("cbeta").ix[0],
            r.atoms.select_atoms("name O").ix[0])
            for r in self.universe.residues
            if (r.resname != "GLY"
                and r.resname in selection.ProteinSelection.prot_res)])
        bonds.extend([
            _
            for s in self.universe.segments
            for _ in zip(s.atoms.select_atoms("name O").ix,
                         s.atoms.select_atoms("name N").ix[1:])
        ])
        self.universe.add_TopologyAttr(Bonds(bonds))


class Polar(Ncsc):
    """Create a universe consisting of the amine, carboxyl, and polar regions.
    """
    model: str = "POLAR"
    describe: str = "c.o.m./c.o.g. of N, C, and polar sidechains of protein"

    def __init__(self,
                 xplor: bool = True,
                 extended: bool = True,
                 com: bool = True,
                 guess_angles: bool = True,
                 cutoff: float = 10.0):
        super().__init__(xplor, extended, com, guess_angles, cutoff)

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
