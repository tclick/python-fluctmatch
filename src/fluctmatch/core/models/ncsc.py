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
"""Class definition for beads using N, carboxyl oxygens, and sidechains."""

from collections import namedtuple
from typing import List, NamedTuple, NoReturn, Tuple

import MDAnalysis as mda
from MDAnalysis.core.topologyattrs import Bonds

from ..base import ModelBase
from ..selection import *


class Model(ModelBase):
    """Universe consisting of the amine, carboxyl, and sidechain regions."""

    model = "NCSC"
    description = "c.o.m./c.o.g. of N, O, and sidechain of protein"

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

        BEADS = namedtuple("BEADS", "N CB O ions")
        self._mapping: NamedTuple = BEADS(
            N="protein and name N",
            CB="hsidechain and not name H*",
            O="protein and name O OT1 OT2 OXT",
            ions="bioion",
        )
        self._selection: NamedTuple = BEADS(
            N="amine", CB="hsidechain", O="carboxyl", ions="bioion"
        )

    def _add_bonds(self) -> NoReturn:
        bonds: List[Tuple[int, int]] = []

        # Create bonds intraresidue atoms
        residues = self.universe.select_atoms("protein").residues
        atom1: AtomGroup = residues.atoms.select_atoms("name N")
        atom2: AtomGroup = residues.atoms.select_atoms("name O")
        bonds.extend(tuple(zip(atom1.ix, atom2.ix)))

        residues = residues.atoms.select_atoms("not resname GLY").residues
        atom1: AtomGroup = residues.atoms.select_atoms("name N")
        atom2: AtomGroup = residues.atoms.select_atoms("name O")
        atom3: AtomGroup = residues.atoms.select_atoms("cbeta")
        bonds.extend(tuple(zip(atom1.ix, atom2.ix)))
        bonds.extend(tuple(zip(atom2.ix, atom3.ix)))

        # Create interresidue bonds
        for segment in self.universe.segments:
            atom1: AtomGroup = segment.atoms.select_atoms("name O")
            atom2: AtomGroup = segment.atoms.select_atoms("name N")
            bonds.extend(tuple(zip(atom1.ix[:-1], atom2.ix[1:])))

        self.universe.add_TopologyAttr(Bonds(bonds))

    def _add_masses(self, universe: mda.Universe) -> NoReturn:
        super()._add_masses(universe)
        amine: AtomGroup = self.universe.select_atoms(self._mapping.N)
        carboxyl: AtomGroup = self.universe.select_atoms(self._mapping.O)
        amine.masses += 0.5 * self.universe.select_atoms("hcalpha").total_mass()
        carboxyl.masses += 0.5 * self.universe.select_atoms("hcalpha").total_mass()

    def _add_charges(self, universe: mda.Universe) -> NoReturn:
        super()._add_charges(universe)
        amine: AtomGroup = self.universe.select_atoms(self._mapping.N)
        carboxyl: AtomGroup = self.universe.select_atoms(self._mapping.O)
        amine.charges += 0.5 * self.universe.select_atoms("hcalpha").total_charge()
        carboxyl.charges += 0.5 * self.universe.select_atoms("hcalpha").total_charge()
