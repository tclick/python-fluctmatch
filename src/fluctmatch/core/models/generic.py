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
"""Model a generic system of all atoms."""
import itertools
from collections import namedtuple
from typing import List, Tuple

import MDAnalysis as mda
from MDAnalysis.coordinates.memory import MemoryReader
from MDAnalysis.topology import guessers

from .. import base
from ..selection import *


class Model(base.ModelBase):
    """Universe consisting of the amine, carboxyl, and sidechain regions."""

    model = "GENERIC"
    description = "all heavy atoms excluding proteins and nucleic acids"

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

        BEAD = namedtuple("BEAD", "bead")
        self._mapping = BEAD(bead="not (protein or nucleic or bioion or water)")
        self._selection = self._mapping

    def create_topology(self, universe: mda.Universe, /) -> None:
        """Determine the topology attributes and initialize the universe.

        Parameters
        ----------
        universe : :class:`~MDAnalysis.Universe`
            An all-atom universe
        """
        atom_group: List[AtomGroup] = [universe.select_atoms(_) for _ in self._mapping]
        self._universe: mda.Universe = mda.Merge(*atom_group)

        float_type = universe.atoms.masses.dtype
        int_type = universe.atoms.resids.dtype

        # Atom
        atomids = np.arange(self._universe.atoms.n_atoms, dtype=int_type)
        attributes = (
            ("radii", np.zeros_like(atomids, dtype=float_type)),
            ("ids", np.zeros_like(atomids, dtype=float_type)),
        )
        for attribute in attributes:
            self._universe.add_TopologyAttr(*attribute)

        self._add_masses(universe)
        self._add_charges(universe)

    def add_trajectory(self, universe: mda.Universe, /) -> None:
        """Add coordinates to the new system.

        Parameters
        ----------
        universe: :class:`~MDAnalysis.Universe`
            An all-atom universe
        """
        if not hasattr(self, "_universe"):
            raise AttributeError(
                "Topologies need to be created before bonds can be added."
            )

        if not hasattr(universe, "trajectory"):
            raise AttributeError(
                "The provided universe does not have coordinates defined."
            )

        selections = itertools.product(
            universe.residues, self._mapping._asdict().items()
        )
        beads: List[AtomGroup] = []
        total_beads: List[AtomGroup] = []
        for residue, (key, selection) in selections:
            value = (
                selection.get(residue.resname)
                if isinstance(selection, dict)
                else selection
            )
            if residue.atoms.select_atoms(value):
                beads.append(residue.atoms.select_atoms(value))

            other_selection = getattr(self._selection, key)
            total_beads.append(residue.atoms.select_atoms(other_selection))

        position_array: List[np.ndarray] = []
        velocity_array: List[np.ndarray] = []
        force_array: List[np.ndarray] = []
        dimension_array: List[np.ndarray] = []
        universe.trajectory.rewind()
        for ts in universe.trajectory:
            dimension_array.append(ts.dimensions)

            # Positions
            try:
                positions = [_.positions for _ in beads if _]
                position_array.append(np.concatenate(positions, axis=0))
            except (AttributeError, mda.NoDataError):
                pass

            # Velocities
            try:
                velocities = [_.velocities.sum(axis=0) for _ in total_beads if _]
                velocity_array.append(np.concatenate(velocities, axis=0))
            except (AttributeError, mda.NoDataError):
                pass

            # Forces
            try:
                forces = [_.forces.sum(axis=0) for _ in total_beads if _]
                force_array.append(np.concatenate(forces, axis=0))
            except (AttributeError, mda.NoDataError):
                pass

        self._universe.trajectory.dimensions_array = np.asarray(dimension_array)
        if self._universe.trajectory.ts.has_positions:
            self._universe.load_new(
                np.asarray(position_array),
                format=MemoryReader,
                dimensions=dimension_array,
            )
        if self._universe.trajectory.ts.has_velocities:
            self._universe.trajectory.velocity_array = np.asarray(velocity_array)
        if self._universe.trajectory.ts.has_forces:
            self._universe.trajectory.force_array = np.asarray(force_array)
        universe.trajectory.rewind()

    def _add_masses(self, universe: mda.Universe, /) -> None:
        selections = itertools.product(universe.residues, self._selection)
        try:
            masses = np.concatenate(
                [
                    residue.atoms.select_atoms(selection).masses
                    for residue, selection in selections
                    if residue.atoms.select_atoms(selection)
                ]
            )
        except (AttributeError, mda.NoDataError):
            masses = np.zeros(
                self._universe.atoms.n_atoms, dtype=universe.atoms.masses.dtype
            )

        self._universe.add_TopologyAttr("masses", masses)

    def _add_charges(self, universe: mda.Universe, /) -> None:
        selections = itertools.product(universe.residues, self._selection)
        try:
            charges = np.concatenate(
                [
                    residue.atoms.select_atoms(selection).charges
                    for residue, selection in selections
                    if residue.atoms.select_atoms(selection)
                ]
            )
        except (AttributeError, mda.NoDataError):
            charges = np.zeros(
                self._universe.atoms.n_atoms, dtype=universe.atoms.masses.dtype
            )

        self._universe.add_TopologyAttr("charges", charges)

    def _add_bonds(self) -> None:
        try:
            atoms: AtomGroup = self._universe.atoms
            positions = self._universe.atoms.positions

            bonds: List[Tuple[int, int]] = guessers.guess_bonds(atoms, positions)
            self._universe.add_TopologyAttr("bonds", bonds)
        except AttributeError:
            pass
