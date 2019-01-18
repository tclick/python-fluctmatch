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
"""Class for elastic network model."""

from typing import List, Optional, Tuple

import MDAnalysis as mda
from MDAnalysis.core.topologyattrs import Atomtypes, Charges, Bonds
from MDAnalysis.lib import distances

from .base import ModelBase, rename_universe
from .selection import *


class Enm(ModelBase):
    """Convert a basic coarse-grain universe into an elastic-network model.

    Determines the interactions between beads via distance cutoffs `rmin` and
    `rmax`. The atoms and residues are also renamed to prevent name collision
    when working with fluctuation matching.

        Parameters
    ----------
    extended : bool, optional
        Renames the residues and atoms according to the extended CHARMM PSF
        format. Standard CHARMM PSF limits the residue and atom names to four
        characters, but the extended CHARMM PSF permits eight characters. The
        residues and atoms are renamed according to the number of segments
        (1: A, 2: B, etc.) and then the residue number or atom index number.
    xplor : bool, optional
        Assigns the atom type as either a numerical or an alphanumerical
        designation. CHARMM normally assigns a numerical designation, but the
        XPLOR version permits an alphanumerical designation with a maximum
        size of 4. The numerical form corresponds to the atom index number plus
        a factor of 100, and the alphanumerical form will be similar the
        standard CHARMM atom name.
    com : bool, optional
        Calculates the bead coordinates using either the center of mass
        (default) or center of geometry.
    guess_angles : bool, optional
        Once Universe has been created, attempt to guess the connectivity
        between atoms.  This will populate the .angles, .dihedrals, and
        .impropers attributes of the Universe.
    cutoff : float, optional
        Used as a bond distance cutoff for an elastic network model; otherwise,
        ignored.
    min_cutoff : float, optional
        Used as a minimum bond distance cutoff for an elastic network model;
        otherwise, ignored.
    charges : bool, optional
        If True, keeps the original charges; otherwise, sets the charges to 0.

    Attributes
    ----------
    universe : :class:`~MDAnalysis.Universe`
        The transformed universe

    """
    model: str = "ENM"
    describe: str = "Elastic network model"

    def __init__(self,
                 xplor: bool = True,
                 extended: bool = True,
                 com: bool = True,
                 guess_angles: bool = False,
                 cutoff: float = 10.0,
                 min_cutoff: Optional[float] = None,
                 charges: bool = False):
        super().__init__(xplor, extended, com, guess_angles, cutoff)

        self._min_cutoff: Optional[float] = min_cutoff
        self._charges: bool = charges

    def create_topology(self, universe: mda.Universe):
        """Deteremine the topology attributes and initialize the universe.

        Parameters
        ----------
        universe : :class:`~MDAnalysis.Universe`
            An all-atom universe
        """
        try:
            self.universe = universe.copy()
        except TypeError:
            self.universe = universe

        rename_universe(self.universe)

        if not self._charges:
            charges = np.zeros(self.universe.atoms.n_atoms)
            self.universe.add_TopologyAttr(Charges(charges))

        atomtypes: np.ndarray = np.arange(self.universe.atoms.n_atoms) + 1
        self.universe.add_TopologyAttr(Atomtypes(atomtypes))

    def add_trajectory(self, universe: mda.Universe):
        pass

    def _add_bonds(self):
        positions: np.ndarray = np.zeros_like(self.universe.atoms.positions)
        for _ in self.universe.trajectory:
            positions += self.universe.atoms.positions
        self.universe.trajectory.rewind()
        positions /= self.universe.trajectory.n_frames

        pairs, _ = distances.self_capped_distance(positions, self._cutoff,
                                                  min_cutoff=self._min_cutoff)
        pairs: List[Tuple[int, int]] = [
            tuple(_)
            for _ in np.unique(pairs, axis=0)
        ]
        bonds: Bonds = Bonds(pairs)
        self.universe.add_TopologyAttr(bonds)
