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
"""Class for elastic network model."""

from typing import ClassVar
from typing import List
from typing import NoReturn
from typing import Tuple

import numpy as np
import MDAnalysis as mda
from MDAnalysis.core.topologyattrs import Angles
from MDAnalysis.core.topologyattrs import Atomtypes
from MDAnalysis.core.topologyattrs import Bonds
from MDAnalysis.core.topologyattrs import Charges
from MDAnalysis.core.topologyattrs import Dihedrals
from MDAnalysis.core.topologyattrs import Impropers
from MDAnalysis.lib import distances

from ..base import ModelBase
from ..base import rename_universe
from ..selection import *


class Model(ModelBase):
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
    description: ClassVar[str] = "Elastic network model"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self._charges: bool = kwargs.pop("charges", False)

    def create_topology(self, universe: mda.Universe) -> NoReturn:
        """Deteremine the topology attributes and initialize the universe.

        Parameters
        ----------
        universe : :class:`~MDAnalysis.Universe`
            An all-atom universe
        """
        try:
            self.universe: mda.Universe = universe.copy()
        except TypeError:
            self.universe: mda.Universe = mda.Universe(
                universe.filename, universe.trajectory.filename
            )

        rename_universe(self.universe)
        n_atoms: int = self.universe.atoms.n_atoms

        if not self._charges:
            charges: np.ndarray = np.zeros(n_atoms)
            self.universe.add_TopologyAttr(Charges(charges))

        self.atomtypes: np.ndarray = np.arange(n_atoms, dtype=int) + 1
        self.universe.add_TopologyAttr(Atomtypes(self.universe.atoms.names))

    def add_trajectory(self, universe: mda.Universe) -> NoReturn:
        pass

    def _add_bonds(self) -> NoReturn:
        # Determine the average positions of the system
        positions: np.ndarray = np.zeros_like(self.universe.atoms.positions)
        for _ in self.universe.trajectory:
            positions += self.universe.atoms.positions
        self.universe.trajectory.rewind()
        positions /= self.universe.trajectory.n_frames

        # Find bonds with distance range of rmin <= r <= rmax
        pairs, _ = distances.self_capped_distance(
            positions, self._rmax, min_cutoff=self._rmin
        )

        # Include predefined bonds
        if hasattr(self.universe, "bonds"):
            bonds: List[Tuple[int, int]] = self.universe.bonds.dump_contents()
            pairs: np.ndarray = np.concatenate([pairs, bonds], axis=0)

        pairs: np.ndarray = np.unique(pairs, axis=0)
        pairs: List[Tuple[int, int]] = list(zip(pairs[:, 0], pairs[:, 1]))
        bonds: Bonds = Bonds(pairs)
        self.universe.add_TopologyAttr(bonds)
        if not self._guess:
            self.universe.add_TopologyAttr(Angles([]))
            self.universe.add_TopologyAttr(Dihedrals([]))
            self.universe.add_TopologyAttr(Impropers([]))
