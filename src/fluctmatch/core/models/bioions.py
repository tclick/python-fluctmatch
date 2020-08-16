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
"""Class defining biological ions."""

from collections import namedtuple
from typing import List, MutableMapping

from MDAnalysis.core.topologyattrs import Atomtypes, Bonds

from ..base import ModelBase
from ..selection import *


class Model(ModelBase):
    """Select ions normally found within biological systems."""

    model = "BIOIONS"
    description = "Common ions found near proteins (Mg Ca Mn Fe Cu Zn Ag)"

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

        BEADS = namedtuple("BEADS", "ions")
        self._guess: bool = False
        self._mapping = BEADS(ions="bioion")
        self._selection = self._mapping

    def _add_atomtypes(self) -> None:
        resnames = np.unique(self._universe.residues.resnames)
        restypes: MutableMapping[str, int] = {
            k: v for k, v in zip(resnames, np.arange(resnames.size) + 20)
        }

        atomtypes: List[int] = [restypes[atom.name] for atom in self._universe.atoms]
        self._universe.add_TopologyAttr(Atomtypes(atomtypes))

    def _add_bonds(self) -> None:
        self._universe.add_TopologyAttr(Bonds([]))
