# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding: utf-8 -*-
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

import abc
import itertools
import logging
import string
from collections import OrderedDict
from typing import List, MutableMapping

import numpy as np
import MDAnalysis as mda
from MDAnalysis.core.topologyattrs import (
    Atomids, Atomnames, Atomtypes, Charges, Masses, Radii, Resids, Resnames,
    Resnums, Segids, TopologyAttr, Angles, Dihedrals, Impropers)
from MDAnalysis.core.topologyobjects import TopologyGroup
from MDAnalysis.coordinates.base import ProtoReader
from MDAnalysis.coordinates.memory import MemoryReader
from MDAnalysis.topology import base as topbase
from MDAnalysis.topology import guessers

from .. import _MODELS, _DESCRIBE
from ..libs.typing import StrMapping

logger: logging.Logger = logging.getLogger(__name__)


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
        Calculates the bead coordinates using either the center of mass (default)
        or center of geometry.
    guess_angles : bool, optional
        Once Universe has been created, attempt to guess the connectivity
        between atoms.  This will populate the .angles, .dihedrals, and
        .impropers attributes of the Universe.
    cutoff : float, optional
        Used as a bond distance cutoff for an elastic network model; otherwise,
        ignored.

    Attributes
    ----------
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
        self._mapping: MutableMapping[str, StrMapping] = OrderedDict()

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
        atomids: List[int] = []
        resids: List[int] = []
        resnames: List[str] = []
        segids: List[str] = []
        charges: List[float] = []
        masses: List[float] = []

        residues: List[mda.AtomGroup] = universe.atoms.split("residue")
        select_residues: enumerate = enumerate(
            itertools.product(residues, self._mapping.items()))
        for i, (res, (name, selection)) in select_residues:
            bead: mda.AtomGroup = res.select_atoms(selection)
            if bead:
                beads.append(bead)
                atomnames.append(name)
                atomids.append(i)
                resids.append(bead.resids[0])
                resnames.append(bead.resnames[0])
                segids.append(bead.segids[0].split("_")[-1])
                try:
                    charges.append(bead.total_charge())
                except AttributeError:
                    charges.append(0.)
                masses.append(bead.total_mass())

        # Atom
        vdwradii: Radii = Radii(np.zeros_like(atomids))
        atomids: Atomids = Atomids(np.asarray(atomids))
        atomnames: Atomnames = Atomnames(
            np.asarray(atomnames, dtype=np.object))
        charges: Charges = Charges(np.asarray(charges))
        masses: Masses = Masses(np.asarray(masses))

        # Residue
        segids: np.ndarray = np.asarray(segids, dtype=np.object)
        resids: np.ndarray = np.asarray(resids, dtype=np.int32)
        resnames: np.ndarray = np.asarray(resnames, dtype=np.object)
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
        attrs: List[TopologyAttr] = [
            atomids, atomnames, charges, masses, vdwradii, residueids,
            residuenums, residuenames, segids
        ]
        for attr in attrs:
            self.universe.add_TopologyAttr(attr)
        self._add_atomtypes()

    def generate_bonds(self):
        """Generate connectivity information for the new system.
        """
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
        univ_traj: ProtoReader = universe.trajectory
        univ_traj.rewind()
        universe.trajectory.rewind()

        residue_selection = itertools.product(universe.residues,
                                              self._mapping.items())
        beads: List[mda.AtomGroup] = []
        for res, (key, selection) in residue_selection:
            if key != "CB":
                beads.append(res.atoms.select_atoms(selection))
            elif key == "CB":
                if isinstance(selection, dict):
                    value = selection.get(res.resname,
                                          "hsidechain and not name H*")
                    beads.append(res.atoms.select_atoms(value))
                else:
                    beads.append(res.atoms.select_atoms(selection))
            beads = [_ for _ in beads if _]

        coordinate_array: List[np.ndarray] = []
        velocity_array: List[np.ndarray] = []
        force_array: List[np.ndarray] = []
        dimensions_array: List[np.ndarray] = []
        for ts in univ_traj:
            dimensions_array.append(ts._unitcell)

            # Positions
            if self.universe.trajectory.ts.has_positions and ts.has_positions:
                coordinates = [
                    bead.center_of_mass()
                    if self._com else bead.center_of_geometry()
                    for bead in beads
                ]
                coordinate_array.append(np.asarray(coordinates))

            # Velocities
            if self.universe.trajectory.ts.has_velocities and ts.has_velocities:
                try:
                    velocities = [bead.velocities for bead in beads]
                    velocity_array.append(np.asarray(velocities))
                except ValueError:
                    pass

            # Forces
            if self.universe.trajectory.ts.has_forces and ts.has_forces:
                try:
                    forces = [bead.velocities for bead in beads]
                    force_array.append(np.asarray(forces))
                except ValueError:
                    pass

        self.universe.trajectory.dimensions_array: np.ndarray = np.asarray(
            dimensions_array)
        if self.universe.trajectory.ts.has_positions:
            coordinate_array: np.ndarray = np.asarray(coordinate_array)
            self.universe.load_new(coordinate_array, format=MemoryReader)
        if self.universe.trajectory.ts.has_velocities:
            self.universe.trajectory.velocity_array: np.ndarray = np.asarray(
                velocity_array)
        if self.universe.trajectory.ts.has_forces:
            self.universe.trajectory.force_array: np.ndarray = np.asarray(
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

    @abc.abstractmethod
    def _add_bonds(self):
        pass

    def _add_angles(self):
        try:
            angles: TopologyGroup = guessers.guess_angles(self.universe.bonds)
            self.universe._topology.add_TopologyAttr(Angles(angles))
            self.universe._generate_from_topology()
        except AttributeError:
            pass

    def _add_dihedrals(self):
        try:
            dihedrals: TopologyGroup = guessers.guess_dihedrals(
                self.universe.angles)
            self.universe._topology.add_TopologyAttr(Dihedrals(dihedrals))
            self.universe._generate_from_topology()
        except AttributeError:
            pass

    def _add_impropers(self):
        try:
            impropers: TopologyGroup = guessers.guess_improper_dihedrals(
                self.universe.angles)
            self.universe._topology.add_TopologyAttr(Impropers(impropers))
            self.universe._generate_from_topology()
        except AttributeError:
            pass


def Merge(*args: List[mda.Universe]) -> mda.Universe:
    """Combine multiple coarse-grain systems into one.

    Parameters
    ----------
    args : iterable of either :class:`~MDAnalysis.Universe` or :class:`~MDAnalysis.AtomGroup`

    Returns
    -------
    :class:`~MDAnalysis.Universe`
        A merged universe.
    """
    logger.warning("This might take a while depending upon the number of "
                   "trajectory frames.")

    # Merge universes
    u: mda.Universe
    universe: mda.Universe = mda.Merge(*[u.atoms for u in args])

    # Merge coordinates
    for u in args:
        u.trajectory.rewind()

    universe1: mda.Universe = args[0]
    universe.trajectory.ts.has_velocities: bool = universe1.trajectory.ts.has_velocities
    universe.trajectory.ts.has_forces: bool = universe1.trajectory.ts.has_forces
    frames: np.ndarray = np.fromiter(
        [u.trajectory.n_frames == universe1.trajectory.n_frames for u in args],
        dtype=bool)
    if not all(frames):
        msg: str = "The trajectories are not the same length."
        logger.error(msg)
        raise ValueError(msg)

    dimensions_array: np.ndarray = (np.mean(
        universe1.trajectory.dimensions_array, axis=0) if hasattr(
            universe1.trajectory, "dimensions_array") else np.asarray(
                [ts.triclinic_dimensions for ts in universe1.trajectory]))

    universe1.universe.trajectory.rewind()
    if universe1.universe.trajectory.n_frames > 1:
        coordinates: List[List[np.ndarray]] = []
        velocities: List[List[np.ndarray]] = []
        forces: List[List[np.ndarray]] = []

        # Accumulate coordinates, velocities, and forces.
        for u in args:
            coordinates.append(
                [ts.positions for ts in u.trajectory if ts.has_positions])
            velocities.append(
                [ts.velocities for ts in u.trajectory if ts.has_velocities])
            forces.append([ts.forces for ts in u.trajectory if ts.has_forces])

        if universe.trajectory.ts.has_positions:
            coordinates: np.ndarray = np.concatenate(coordinates, axis=1)
            if universe.atoms.n_atoms != coordinates.shape[1]:
                msg = ("The number of sites does not match the number of "
                       "coordinates.")
                logger.error(msg)
                raise RuntimeError(msg)
            n_frames, n_beads, _ = coordinates.shape
            logger.info(f"The new universe has {n_beads:d} beads in "
                        f"{n_frames:d} frames.")

            universe.load_new(coordinates, format=MemoryReader)
            universe.trajectory.dimensions_array = dimensions_array.copy()
        if universe.trajectory.ts.has_velocities:
            velocities: np.ndarray = np.concatenate(velocities, axis=1)
            universe.trajectory.velocity_array: np.ndarray = velocities.copy()
        if universe.trajectory.ts.has_forces:
            forces: np.ndarray = np.concatenate(forces, axis=1)
            universe.trajectory.force_array: np.ndarray = forces.copy()

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

    universe._topology.add_TopologyAttr(Atomnames(atomnames))
    universe._topology.add_TopologyAttr(Resnames(resnames))
    if not np.issubdtype(universe.atoms.types.dtype, np.int64):
        universe._topology.add_TopologyAttr(Atomtypes(atomnames))
    universe._generate_from_topology()
