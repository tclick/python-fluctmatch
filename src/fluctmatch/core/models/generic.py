# ------------------------------------------------------------------------------
#   python-fluctmatch
#   Copyright (c) 2013-2019 Timothy H. Click, Ph.D.
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
#   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS “AS IS”
#   AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
#   IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
#   ARE DISCLAIMED. IN NO EVENT SHALL THE REGENTS OR CONTRIBUTORS BE LIABLE FOR
#   ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
#   DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
#   SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
#   CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
#   OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
#   OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
#   Timothy H. Click, Nixon Raj, and Jhih-Wei Chu.
#   Simulation. Meth Enzymology. 578 (2016), 327-342,
#   Calculation of Enzyme Fluctuograms from All-Atom Molecular Dynamics
#   doi:10.1016/bs.mie.2016.05.024.
#
# ------------------------------------------------------------------------------

#
#  All rights reserved.
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:
#
#
#
#  Neither the name of the author nor the names of its contributors may be used
#  to endorse or promote products derived from this software without specific
#  prior written permission.
#
"""Model a generic system of all atoms."""

from typing import List
from typing import NoReturn
from typing import Tuple

import MDAnalysis as mda
from MDAnalysis.coordinates.memory import MemoryReader
from MDAnalysis.core.topologyattrs import Bonds
from MDAnalysis.topology import guessers

from .. import base
from ..selection import *


class Model(base.ModelBase):
    """Universe consisting of the amine, carboxyl, and sidechain regions."""

    description: ClassVar[str] = (
        "all heavy atoms excluding proteins and nucleic acids"
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def create_topology(self, universe: mda.Universe) -> NoReturn:
        """Determine the topology attributes and initialize the universe.

        Parameters
        ----------
        universe : :class:`~MDAnalysis.Universe`
            An all-atom universe
        """
        self._mapping: str = "not (protein or nucleic or bioion or water)"
        ag: mda.AtomGroup = universe.select_atoms(self._mapping)
        self.universe: mda.Universe = mda.Merge(ag)

    def add_trajectory(self, universe: mda.Universe) -> NoReturn:
        """Add coordinates to the new system.

        Parameters
        ----------
        universe: :class:`~MDAnalysis.Universe`
            An all-atom universe
        """
        if not hasattr(self, "universe"):
            raise AttributeError(
                "Topologies need to be created before bonds " "can be added."
            )

        if not hasattr(universe, "trajectory"):
            raise AttributeError(
                "The provided universe does not have " "coordinates defined."
            )

        ag: mda.AtomGroup = universe.select_atoms(self._mapping)
        trajectory = universe.trajectory
        trajectory.rewind()

        trajectory2 = self.universe.trajectory
        trajectory2.ts.has_positions = trajectory.ts.has_positions
        trajectory2.ts.has_velocities = trajectory.ts.has_velocities
        trajectory2.ts.has_forces = trajectory.ts.has_forces

        position_array: List[np.ndarray] = []
        velocity_array: List[np.ndarray] = []
        force_array: List[np.ndarray] = []
        dimension_array: List[np.ndarray] = []

        for ts in trajectory:
            dimension_array.append(ts.dimensions)

            # Positions
            try:
                position_array.append(ag.positions)
            except (AttributeError, mda.NoDataError):
                pass

            # Velocities
            try:
                velocity_array.append(ag.velocities)
            except (AttributeError, mda.NoDataError):
                pass

            # Forces
            try:
                force_array.append(ag.forces)
            except (AttributeError, mda.NoDataError):
                pass

        trajectory2.dimensions_array = np.asarray(dimension_array)
        if trajectory2.ts.has_positions:
            position_array: np.ndarray = np.asarray(position_array)
            self.universe.load_new(
                position_array, format=MemoryReader, dimensions=dimension_array
            )
        if trajectory2.ts.has_velocities:
            trajectory2.velocity_array = np.asarray(velocity_array)
        if trajectory2.ts.has_forces:
            trajectory2.force_array = np.asarray(force_array)
        universe.trajectory.rewind()

    def _add_bonds(self) -> NoReturn:
        try:
            atoms: mda.AtomGroup = self.universe.atoms
            positions: np.ndarray = self.universe.atoms.positions

            bonds: List[Tuple[int, int]] = guessers.guess_bonds(atoms,
                                                                positions)
            self.universe.add_TopologyAttr(Bonds(bonds))
        except AttributeError:
            pass
