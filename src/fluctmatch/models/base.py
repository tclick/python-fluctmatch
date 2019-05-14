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
#  Please cite your use of fluctmatch in published work:
#
#  Timothy H. Click, Nixon Raj, and Jhih-Wei Chu.
#  Calculation of Enzyme Fluctuograms from All-Atom Molecular Dynamics
#  Simulation. Meth Enzymology. 578 (2016), 327-342,
#  doi:10.1016/bs.mie.2016.05.024.
"""Base class for all models."""

import abc
import itertools
import logging
import string
from collections import OrderedDict
from typing import Generator, Iterable, List, MutableMapping, TypeVar

import numpy as np
import MDAnalysis as mda
from MDAnalysis.core.topologyattrs import (
    Atomids, Atomnames, Atomtypes, Charges, Masses, Radii, Resids, Resnames,
    Resnums, Segids, TopologyAttr, Angles, Dihedrals, Impropers)
from MDAnalysis.core.topologyobjects import TopologyGroup
from MDAnalysis.coordinates.memory import MemoryReader
from MDAnalysis.topology import base as topbase
from MDAnalysis.topology import guessers

from .. import _MODELS, _DESCRIBE
from ..libs.typing import StrMapping

logger: logging.Logger = logging.getLogger(__name__)
MDUniverse = TypeVar("MDUniverse", mda.Universe, Iterable[mda.Universe])


class ModelBase(abc.ABC):
    """Base class for creating coarse-grain models.

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

    Attributes
    ----------
    universe : :class:`~MDAnalysis.Universe`
        The transformed universe
    """
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        _MODELS[cls.model.upper()]: object = cls
        _DESCRIBE[cls.model.upper()]: str = cls.describe

    def __init__(self,
                 xplor: bool = True,
                 extended: bool = True,
                 com: bool = True,
                 guess_angles: bool = True,
                 cutoff: float = 10.0):
        """Initialise like a normal MDAnalysis Universe but give the mapping and
        com keywords.

        Mapping must be a dictionary with atom names as keys.
        Each name must then correspond to a selection string,
        signifying how to split up a single residue into many beads.
        eg:
        mapping = {"CA":"protein and name CA",
                   "CB":"protein and not name N HN H HT* H1 H2 H3 CA HA* C O OXT
                   OT*"}
        would split residues into 2 beads containing the C-alpha atom and the
        sidechain.
        """
        # Coarse grained Universe
        # Make a blank Universe for myself.
        super().__init__()

        self.universe: mda.Universe = None
        self._xplor: bool = xplor
        self._extended: bool = extended
        self._com: bool = com
        self._guess: bool = guess_angles
        self._cutoff: float = cutoff

        # Dictionary for specific bead selection
        self._mapping: MutableMapping[str, StrMapping] = OrderedDict()

        # Dictionary for all-atom to bead selection. This is particularly useful
        # for mass and charge topology attributes.
        self._selection: MutableMapping[str, StrMapping] = {}

    def create_topology(self, universe: mda.Universe):
        """Deteremine the topology attributes and initialize the universe.
        
        Parameters
        ----------
        universe : :class:`~MDAnalysis.Universe`
            An all-atom universe
        """
        # Allocate arrays
        beads: List[mda.AtomGroup] = []
        atomnames: List[str] = []
        selections: Generator = itertools.product(universe.residues,
                                                  self._mapping.items())

        for residue, (name, selection) in selections:
            if isinstance(selection, dict):
                value: mda.AtomGroup = selection.get(
                    residue.resname, "hsidechain and not name H*")
                bead: mda.AtomGroup = residue.atoms.select_atoms(value)
            else:
                bead: mda.AtomGroup = residue.atoms.select_atoms(selection)
            if bead:
                beads.append(bead)
                atomnames.append(name)

        atomids: np.ndarray = np.arange(len(beads), dtype=int)

        # Atom
        vdwradii: Radii = Radii(np.zeros_like(atomids, dtype=np.float32))
        atomids: Atomids = Atomids(atomids)
        atomnames: Atomnames = Atomnames(atomnames)

        # Residue
        resids: np.ndarray = np.asarray([bead.resids[0] for bead in beads],
                                        dtype=int)
        resnames: np.ndarray = np.asarray([bead.resnames[0] for bead in beads],
                                          dtype=object)
        segids: np.ndarray = np.asarray([bead.segids[0].split("_")[-1]
                                         for bead in beads], dtype=object)
        residx, (new_resids, new_resnames,
                 perres_segids) = topbase.change_squash(
                     (resids, resnames, segids), (resids, resnames, segids))

        # transform from atom:Rid to atom:Rix
        residueids: Resids = Resids(new_resids)
        residuenums: Resnums = Resnums(new_resids.copy())
        residuenames: Resnames = Resnames(new_resnames)

        # Segment
        segidx, perseg_segids = topbase.squash_by(perres_segids)[:2]
        segids: Segids = Segids(perseg_segids)

        # Create universe and add attributes
        self.universe: mda.Universe = mda.Universe.empty(
            len(atomids),
            n_residues=len(new_resids),
            n_segments=len(segids),
            atom_resindex=residx,
            residue_segindex=segidx,
            trajectory=True)

        # Add additonal attributes
        attrs: List[TopologyAttr] = [
            atomids, atomnames, vdwradii, residueids,
            residuenums, residuenames, segids
        ]
        for attr in attrs:
            self.universe.add_TopologyAttr(attr)
        self._add_atomtypes()
        self._add_masses(universe)
        self._add_charges(universe)

    def generate_bonds(self):
        """Generate connectivity information for the new system."""
        if not hasattr(self, "universe"):
            raise AttributeError("Topologies need to be created before bonds "
                                 "can be added.")
        self._add_bonds()
        if self._guess:
            self._add_angles()
            self._add_dihedrals()
            self._add_impropers()

    def add_trajectory(self, universe: mda.Universe):
        """Add coordinates to the new system.
        
        Parameters
        ----------
        universe: :class:`~MDAnalysis.Universe`
            An all-atom universe
        """
        if not hasattr(self, "universe"):
            raise AttributeError("Topologies need to be created before bonds "
                                 "can be added.")

        if not hasattr(universe, "trajectory"):
            raise AttributeError("The provided universe does not have "
                                 "coordinates defined.")
        trajectory = universe.trajectory
        trajectory.rewind()

        selections: Generator = itertools.product(universe.residues,
                                                  self._mapping.values())
        beads: List[mda.AtomGroup] = []
        for residue, selection in selections:
            if isinstance(selection, dict):
                value: mda.AtomGroup = selection.get(
                    residue.resname, "hsidechain and not name H*")
                bead: mda.AtomGroup = residue.atoms.select_atoms(value)
            else:
                bead: mda.AtomGroup = residue.atoms.select_atoms(selection)
            beads.append(bead)

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
                positions = [
                    bead.center_of_mass()
                    if self._com
                    else bead.center_of_geometry()
                    for bead in beads
                    if bead
                ]
                position_array.append(np.asarray(positions))
            except (AttributeError, mda.NoDataError):
                pass

            # Velocities
            try:
                velocities = [bead.velocities for bead in beads if bead]
                velocity_array.append(np.asarray(velocities))
            except (AttributeError, mda.NoDataError):
                pass

            # Forces
            try:
                forces = [bead.velocities for bead in beads if bead]
                force_array.append(np.asarray(forces))
            except (AttributeError, mda.NoDataError):
                pass

        trajectory2.dimensions_array: np.ndarray = np.asarray(dimension_array)
        if trajectory2.ts.has_positions:
            position_array: np.ndarray = np.asarray(position_array)
            self.universe.load_new(position_array, format=MemoryReader,
                                   dimensions=dimension_array)
        if trajectory2.ts.has_velocities:
            trajectory2.velocity_array: np.ndarray = np.asarray(
                velocity_array)
        if trajectory2.ts.has_forces:
            trajectory2.force_array: np.ndarray = np.asarray(
                force_array)
        universe.trajectory.rewind()

    def transform(self, universe: mda.Universe) -> mda.Universe:
        """Convert an all-atom universe to a coarse-grain model.

        Topologies are generated, bead connections are determined, and positions
        are read. This is a wrapper for the other three methods.
        
        Parameters
        ----------
        universe: :class:`~MDAnalysis.Universe`
            An all-atom universe

        Returns
        -------
        A coarse-grain model
        """
        self.create_topology(universe)
        self.generate_bonds()
        self.add_trajectory(universe)
        return self.universe

    def _add_atomtypes(self):
        n_atoms: int = self.universe.atoms.n_atoms
        atomtypes: Atomtypes = Atomtypes(np.arange(n_atoms) + 100)
        self.universe.add_TopologyAttr(atomtypes)

    def _add_masses(self, universe: mda.Universe):
        residues: List[mda.AtomGroup] = universe.atoms.split("residue")
        select_residues: Generator = itertools.product(residues,
                                                       self._selection.values())

        try:
            masses: np.ndarray = np.fromiter([
                res.select_atoms(selection).total_mass()
                for res, selection in select_residues
                if res.select_atoms(selection)
            ], dtype=np.float32)
        except (AttributeError, mda.NoDataError):
            masses: np.ndarray = np.zeros(self.universe.atoms.n_atoms,
                                          dtype=np.float32)

        self.universe.add_TopologyAttr(Masses(masses))

    def _add_charges(self, universe: mda.Universe):
        residues: List[mda.AtomGroup] = universe.atoms.split("residue")
        select_residues: Generator = itertools.product(residues,
                                                       self._selection.values())

        try:
            charges: np.ndarray = np.fromiter([
                res.select_atoms(selection).total_charge()
                for res, selection in select_residues
                if res.select_atoms(selection)
            ], dtype=np.float32)
        except (AttributeError, mda.NoDataError):
            charges: np.ndarray = np.zeros(self.universe.atoms.n_atoms,
                                           dtype=np.float32)

        self.universe.add_TopologyAttr(Charges(charges))

    @abc.abstractmethod
    def _add_bonds(self):
        pass

    def _add_angles(self):
        try:
            angles: TopologyGroup = guessers.guess_angles(self.universe.bonds)
            self.universe.add_TopologyAttr(Angles(angles))
        except AttributeError:
            pass

    def _add_dihedrals(self):
        try:
            dihedrals: TopologyGroup = guessers.guess_dihedrals(
                self.universe.angles)
            self.universe.add_TopologyAttr(Dihedrals(dihedrals))
        except AttributeError:
            pass

    def _add_impropers(self):
        try:
            impropers: TopologyGroup = guessers.guess_improper_dihedrals(
                self.universe.angles)
            self.universe.add_TopologyAttr(Impropers(impropers))
        except AttributeError:
            pass


def Merge(*args: MDUniverse) -> mda.Universe:
    """Combine multiple coarse-grain systems into one.

    Parameters
    ----------
    args : iterable of :class:`~MDAnalysis.Universe`

    Returns
    -------
    :class:`~MDAnalysis.Universe`
        A merged universe.
    """
    logger.warning("This might take a while depending upon the number of "
                   "trajectory frames.")

    # Merge universes
    universe: mda.Universe = mda.Merge(*[u.atoms for u in args])
    trajectory: mda._READERS[universe.trajectory.format] = universe.trajectory

    # Merge coordinates
    for u in args:
        u.trajectory.rewind()

    universe1: mda.Universe = args[0]
    trajectory1 = universe1.trajectory
    trajectory.ts.has_velocities: bool = (
        trajectory1.ts.has_velocities
    )
    trajectory.ts.has_forces: bool = trajectory1.ts.has_forces
    frames: np.ndarray = np.fromiter(
        [u.trajectory.n_frames == trajectory1.n_frames for u in args],
        dtype=bool)
    if not all(frames):
        msg: str = "The trajectories are not the same length."
        logger.error(msg)
        raise ValueError(msg)

    dimensions: np.ndarray = (
        trajectory1.dimensions_array
        if hasattr(trajectory1, "dimensions_array")
        else np.asarray([ts.dimensions for ts in trajectory1]))

    trajectory1.rewind()
    if trajectory1.n_frames > 1:
        positions: List[List[np.ndarray]] = []
        velocities: List[List[np.ndarray]] = []
        forces: List[List[np.ndarray]] = []

        # Accumulate coordinates, velocities, and forces.
        for u in args:
            positions.append(
                [ts.positions for ts in u.trajectory if ts.has_positions])
            velocities.append(
                [ts.velocities for ts in u.trajectory if ts.has_velocities])
            forces.append([ts.forces for ts in u.trajectory if ts.has_forces])

        if trajectory.ts.has_positions:
            positions: np.ndarray = np.concatenate(positions, axis=1)
            if universe.atoms.n_atoms != positions.shape[1]:
                msg = ("The number of sites does not match the number of "
                       "coordinates.")
                logger.error(msg)
                raise RuntimeError(msg)
            n_frames, n_beads, _ = positions.shape
            logger.info(f"The new universe has {n_beads:d} beads in "
                        f"{n_frames:d} frames.")
            universe.load_new(positions, format=MemoryReader,
                              dimensions=dimensions)

        if trajectory.ts.has_velocities:
            velocities: np.ndarray = np.concatenate(velocities, axis=1)
            trajectory.velocity_array: np.ndarray = velocities.copy()
        if trajectory.ts.has_forces:
            forces: np.ndarray = np.concatenate(forces, axis=1)
            trajectory.force_array: np.ndarray = forces.copy()

    return universe


def rename_universe(universe: mda.Universe):
    """Rename the atoms and residues within a universe.

    Standardizes naming of the universe by renaming atoms and residues based
    upon the number of segments. Atoms are labeled as 'A001', 'A002', 'A003',
    ..., 'A999' for the first segment, and 'B001', 'B002', 'B003', ..., 'B999'
    for the second segment. Residues are named in a similar fashion according to
    their segment.

    Parameters
    ----------
    universe : :class:`~MDAnalysis.Universe`
        A collection of atoms in a universe.
    """
    logger.info("Renaming atom names and atom types within the universe.")
    atomnames: np.ndarray = np.array([
        "{}{:0>3d}".format(lett, i)
        for lett, segment in zip(string.ascii_uppercase, universe.segments)
        for i, _ in enumerate(segment.atoms, 1)
    ])
    resnames: np.ndarray = np.array([
        "{}{:0>3d}".format(lett, i)
        for lett, segment in zip(string.ascii_uppercase, universe.segments)
        for i, _ in enumerate(segment.residues, 1)
    ])

    universe.add_TopologyAttr(Atomnames(atomnames))
    universe.add_TopologyAttr(Resnames(resnames))
    if not np.issubdtype(universe.atoms.types.dtype, np.int64):
        universe.add_TopologyAttr(Atomtypes(atomnames))
