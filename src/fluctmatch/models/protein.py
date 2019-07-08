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
"""Classes for various protein models."""

from typing import ClassVar
from typing import List
from typing import Mapping
from typing import Tuple

from MDAnalysis.core.topologyattrs import Bonds

from .base import ModelBase


class Calpha(ModelBase):
    """Universe defined by the protein C-alpha."""

    model: ClassVar[str] = "CALPHA"
    describe: ClassVar[str] = "C-alpha of a protein"

    def __init__(
        self,
        xplor: bool = True,
        extended: bool = True,
        com: bool = True,
        guess_angles: bool = True,
        cutoff: float = 10.0,
    ):
        super().__init__(xplor, extended, com, guess_angles, cutoff)

        self._mapping: Mapping[str, str] = dict(CA="calpha", ions="bioion")
        self._selection: Mapping[str, str] = dict(CA="protein", ions="bioion")

    def _add_bonds(self):
        bonds: List[Tuple[int, int]] = []
        bonds.extend(
            [
                idx
                for segment in self.universe.segments
                for idx in zip(
                    segment.atoms.select_atoms("calpha").ix,
                    segment.atoms.select_atoms("calpha").ix[1:],
                )
            ]
        )
        self.universe.add_TopologyAttr(Bonds(bonds))


class Caside(ModelBase):
    """Universe consisting of the C-alpha and sidechains of a protein."""

    model: ClassVar[str] = "CASIDE"
    describe: ClassVar[str] = "C-alpha and sidechain (c.o.m./c.o.g.) of protein"

    def __init__(
        self,
        xplor: bool = True,
        extended: bool = True,
        com: bool = True,
        guess_angles: bool = True,
        cutoff: float = 10.0,
    ):
        super().__init__(xplor, extended, com, guess_angles, cutoff)

        self._mapping: Mapping[str, str] = dict(
            CA="calpha", CB="hsidechain and not name H*", ions="bioion"
        )
        self._selection: Mapping[str, str] = dict(
            CA="hbackbone", CB="hsidechain", ions="bioion"
        )

    def _add_bonds(self):
        bonds: List[Tuple[int, int]] = []
        bonds.extend(
            [
                idx
                for segment in self.universe.segments
                for idx in zip(
                    segment.atoms.select_atoms("calpha").ix,
                    segment.atoms.select_atoms("calpha").ix[1:],
                )
            ]
        )
        bonds.extend(
            [
                (
                    residue.atoms.select_atoms("calpha").ix[0],
                    residue.atoms.select_atoms("cbeta").ix[0],
                )
                for residue in self.universe.select_atoms("protein").residues
                if residue.resname != "GLY"
            ]
        )
        self.universe.add_TopologyAttr(Bonds(bonds))


class Ncsc(ModelBase):
    """Universe consisting of the amine, carboxyl, and sidechain regions."""

    model: ClassVar[str] = "NCSC"
    describe: ClassVar[str] = "c.o.m./c.o.g. of N, O, and sidechain of protein"

    def __init__(
        self,
        xplor: bool = True,
        extended: bool = True,
        com: bool = True,
        guess_angles: bool = True,
        cutoff: float = 10.0,
    ):
        super().__init__(xplor, extended, com, guess_angles, cutoff)

        self._mapping: Mapping[str, str] = dict(
            N="protein and name N",
            CB="hsidechain and not name H*",
            O="protein and name O OT1 OT2 OXT",
            ions="bioion",
        )
        self._selection: Mapping[str, str] = dict(
            N="amine", CB="hsidechain", O="carboxyl", ions="bioion"
        )

    def _add_bonds(self):
        bonds: List[Tuple[int, int]] = []
        bonds.extend(
            [
                idx
                for segment in self.universe.segments
                for idx in zip(
                    segment.atoms.select_atoms("name N").ix,
                    segment.atoms.select_atoms("name O").ix,
                )
            ]
        )
        bonds.extend(
            [
                (
                    residue.atoms.select_atoms("name N").ix[0],
                    residue.atoms.select_atoms("cbeta").ix[0],
                )
                for residue in self.universe.select_atoms("protein").residues
                if residue.resname != "GLY"
            ]
        )
        bonds.extend(
            [
                (
                    residue.atoms.select_atoms("cbeta").ix[0],
                    residue.atoms.select_atoms("name O").ix[0],
                )
                for residue in self.universe.select_atoms("protein").residues
                if residue.resname != "GLY"
            ]
        )
        bonds.extend(
            [
                idx
                for segment in self.universe.segments
                for idx in zip(
                    segment.atoms.select_atoms("name O").ix,
                    segment.atoms.select_atoms("name N").ix[1:],
                )
            ]
        )
        self.universe.add_TopologyAttr(Bonds(bonds))


class Polar(Ncsc):
    """Universe consisting of the amine, carboxyl, and polar regions."""

    model: ClassVar[str] = "POLAR"
    describe: ClassVar[str] = "c.o.m./c.o.g. of N, C, and polar sidechains of protein"

    def __init__(
        self,
        xplor: bool = True,
        extended: bool = True,
        com: bool = True,
        guess_angles: bool = True,
        cutoff: float = 10.0,
    ):
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
