# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding: utf-8 -*-
# vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4
#
# fluctmatch --- https://github.com/tclick/python-fluctmatch
# Copyright (c) 2013-2017 The fluctmatch Development Team and contributors
# (see the file AUTHORS for the full list of names)
#
# Released under the New BSD license.
#
# Please cite your use of fluctmatch in published work:
#
# Timothy H. Click, Nixon Raj, and Jhih-Wei Chu.
# Calculation of Enzyme Fluctuograms from All-Atom Molecular Dynamics
# Simulation. Meth Enzymology. 578 (2016), 327-342,
# doi:10.1016/bs.mie.2016.05.024.
#
# The original code is from Richard J. Gowers.
# https://github.com/richardjgowers/MDAnalysis-coarsegraining
#
import abc
import itertools
import logging
import string
import traceback
from typing import Dict, List

import numpy as np
import MDAnalysis as mda
from MDAnalysis.core.topologyattrs import (
    Atomids, Atomnames, Atomtypes, Charges, Masses, Radii, Resids,
    Resnames, Resnums, Segids, TopologyAttr, Angles, Dihedrals, Impropers)
from MDAnalysis.core.topologyobjects import TopologyGroup
from MDAnalysis.coordinates.base import ProtoReader
from MDAnalysis.coordinates.memory import MemoryReader
from MDAnalysis.topology import base as topbase
from MDAnalysis.topology import guessers

from .. import _MODELS, _DESCRIBE
from ..libs.typing import FileName, MDUniverse
from . import trajectory

logger: logging.Logger = logging.getLogger(__name__)


class ModelBase(abc.ABC):
    """Base class for creating coarse-grain models.

    Parameters
    ----------
    topology : filename or Topology object
        A CHARMM/XPLOR PSF topology file, PDB file or Gromacs GRO file; used to
        define the list of atoms. If the file includes bond information,
        partial charges, atom masses, ... then these data will be available to
        MDAnalysis. A "structure" file (PSF, PDB or GRO, in the sense of a
        topology) is always required. Alternatively, an existing
        :class:`MDAnalysis.core.topology.Topology` instance may also be given.
    extended
        Renames the residues and atoms according to the extended CHARMM PSF
        format. Standard CHARMM PSF limits the residue and atom names to four
        characters, but the extended CHARMM PSF permits eight characters. The
        residues and atoms are renamed according to the number of segments
        (1: A, 2: B, etc.) and then the residue number or atom index number.
     xplor
        Assigns the atom type as either a numerical or an alphanumerical
        designation. CHARMM normally assigns a numerical designation, but the
        XPLOR version permits an alphanumerical designation with a maximum
        size of 4. The numerical form corresponds to the atom index number plus
        a factor of 100, and the alphanumerical form will be similar the
        standard CHARMM atom name.
    topology_format
        Provide the file format of the topology file; ``None`` guesses it from
        the file extension [``None``] Can also pass a subclass of
        :class:`MDAnalysis.topology.base.TopologyReaderBase` to define a custom
        reader to be used on the topology file.
    format
        Provide the file format of the coordinate or trajectory file; ``None``
        guesses it from the file extension. Note that this keyword has no
        effect if a list of file names is supplied because the "chained" reader
        has to guess the file format for each individual list member.
        [``None``] Can also pass a subclass of
        :class:`MDAnalysis.coordinates.base.ProtoReader` to define a custom
        reader to be used on the trajectory file.
    guess_bonds : bool, optional
        Once Universe has been loaded, attempt to guess the connectivity
        between atoms.  This will populate the .bonds .angles and .dihedrals
        attributes of the Universe.
    vdwradii : Dict, optional
        For use with *guess_bonds*. Supply a dict giving a vdwradii for each
        atom type which are used in guessing bonds.
    is_anchor : bool, optional
        When unpickling instances of
        :class:`MDAnalysis.core.groups.AtomGroup` existing Universes are
        searched for one where to anchor those atoms. Set to ``False`` to
        prevent this Universe from being considered. [``True``]
    anchor_name : str, optional
        Setting to other than ``None`` will cause
        :class:`MDAnalysis.core.groups.AtomGroup` instances pickled from the
        Universe to only unpickle if a compatible Universe with matching
        *anchor_name* is found. Even if *anchor_name* is set *is_anchor* will
        still be honored when unpickling.
    in_memory
        After reading in the trajectory, transfer it to an in-memory
        representations, which allow for manipulation of coordinates.
    in_memory_step
        Only read every nth frame into in-memory representation.

    Attributes
    ----------
    trajectory
        currently loaded trajectory reader;
    dimensions
        current system dimensions (simulation unit cell, if set in the
        trajectory)
    atoms, residues, segments
        master Groups for each topology level
    bonds, angles, dihedrals
        master ConnectivityGroups for each connectivity type
    """
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        _MODELS[cls.model.upper()]: object = cls
        _DESCRIBE[cls.model.upper()]: str = cls.describe


    def __init__(self, xplor: bool=True, extended: bool=True, com: bool=True,
                 guess_angles: bool=True, cutoff: float=10.0):
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

        self.xplor: bool = xplor
        self.extended: bool = extended
        self.com: bool = com
        self.guess: bool = guess_angles
        self.cutoff: float = cutoff
        self.mapping: dict = {}

    def _initialize(self, *args, **kwargs):
        try:
            mapping: Dict = kwargs.pop("mapping")
        except KeyError:
            raise ValueError("CG mapping has not been defined.")

        # Fake up some beads
        self._topology = self._apply_map(mapping)
        self._generate_from_topology()
        self._add_bonds()
        if kwargs.get("guess_angles", True):
            self._add_angles()
            self._add_dihedrals()
            self._add_impropers()

        # This replaces load_new in a traditional Universe
        try:
            self.trajectory = trajectory._Trajectory(
                self.atu, mapping, n_atoms=self.atoms.n_atoms, com=self._com
            )
        except (IOError, TypeError) as exc:
            tb: List[str] = traceback.format_exc()
            RuntimeError(
                f"Unable to open {self.atu.trajectory.filename}"
            ).with_traceback(tb)

    def create_topology(self, universe: mda.Universe):
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
            itertools.product(residues, self.mapping.items()))
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

        beads: np.ndarray = np.array(beads)
        n_atoms: int = len(beads)

        # Atom
        vdwradii: Radii = Radii(np.zeros_like(atomids))
        atomids: Atomids = Atomids(np.asarray(atomids))
        atomnames: Atomnames = Atomnames(np.asarray(atomnames, dtype=np.object))
        atomtypes: Atomtypes = Atomtypes(np.asarray(np.arange(n_atoms) + 100))
        charges: Charges = Charges(np.asarray(charges))
        masses: Masses = Masses(np.asarray(masses))

        # Residue
        # resids, resnames
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

        attrs: List[TopologyAttr] = [atomids, atomnames, atomtypes,
                                     charges, masses, vdwradii,
                                     residueids, residuenums, residuenames,
                                     segids]
        self.universe: mda.Universe = mda.Universe.empty(
            len(atomids), n_residues=len(new_resids), n_segments=len(segids),
            atom_resindex=residx, residue_segindex=segidx, trajectory=True)
        for attr in attrs:
            self.universe.add_TopologyAttr(attr)

        self._set_masses()
        self._set_charges()

    def add_bonds(self):
        if not hasattr(self, "universe"):
            raise AttributeError("Topologies need to be created before bonds "
                                 "can be added.")
        self._add_bonds()
        if self.guess:
            self._add_angles()
            self._add_dihedrals()
            self._add_impropers()

    def add_trajectory(self, universe: mda.Universe):
        if not hasattr(self, "universe"):
            raise AttributeError("Topologies need to be created before bonds "
                                 "can be added.")

        if not hasattr(universe, "trajectory"):
            raise AttributeError("The provided universe does not have "
                                 "coordinates defined.")
        univ_traj: ProtoReader = universe.trajectory
        univ_traj.rewind()

        residue_selection = itertools.product(universe.residues,
                                              self.mapping.items())
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
            dimensions_array.append(ts.unitcell)
            if self.universe.trajectory.ts.has_positions and ts.has_positions:
                coordinates = [
                    bead.center_of_mass()
                    if self.com
                    else bead.center_of_geometry()
                    for bead in beads
                ]
                coordinate_array.append(np.asarray(coordinates))

            if self.universe.trajectory.ts.has_velocities and ts.has_velocities:
                try:
                    velocities = [bead.velocities for bead in beads]
                    velocity_array.append(np.asarray(velocities))
                except ValueError:
                    pass

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
                self.universe.angles
            )
            self.universe._topology.add_TopologyAttr(Dihedrals(dihedrals))
            self.universe._generate_from_topology()
        except AttributeError:
            pass

    def _add_impropers(self):
        try:
            impropers: TopologyGroup = guessers.guess_improper_dihedrals(
                self.universe.angles
            )
            self.universe._topology.add_TopologyAttr(Impropers(impropers))
            self.universe._generate_from_topology()
        except AttributeError:
            pass

    @abc.abstractmethod
    def _set_masses(self):
        pass

    @abc.abstractmethod
    def _set_charges(self):
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
    u: MDUniverse
    universe: mda.Universe = mda.Merge(*[u.atoms for u in args])

    # Merge coordinates
    for u in args:
        u.trajectory.rewind()

    universe1: mda.Universe = args[0]
    universe.trajectory.ts.has_velocities: bool = universe1.trajectory.ts.has_velocities
    universe.trajectory.ts.has_forces: bool = universe1.trajectory.ts.has_forces
    frames: np.ndarray = np.fromiter([
        u.trajectory.n_frames == universe1.trajectory.n_frames
        for u in args], dtype=bool)
    if not all(frames):
        msg: str = "The trajectories are not the same length."
        logger.error(msg)
        raise ValueError(msg)

    dimensions_array: np.ndarray = (
        np.mean(universe1.trajectory.dimensions_array, axis=0)
        if hasattr(universe1.trajectory, "dimensions_array")
        else np.asarray([ts.triclinic_dimensions for ts in universe1.trajectory]))

    universe1.universe.trajectory.rewind()
    if universe1.universe.trajectory.n_frames > 1:
        coordinates: List[np.ndarray] = []
        velocities: List[np.ndarray] = []
        forces: List[np.ndarray] = []

        # Accumulate coordinates, velocities, and forces.
        for u in args:
            coordinates.append([
                ts.positions
                for ts in u.trajectory
                if ts.has_positions])
            velocities.append([
                ts.velocities
                for ts in u.trajectory
                if ts.has_velocities])
            forces.append([
                ts.forces
                for ts in u.trajectory
                if ts.has_forces])

        if universe.trajectory.ts.has_positions:
            coordinates: np.ndarray = np.concatenate(coordinates, axis=1)
            if universe.atoms.n_atoms != coordinates.shape[1]:
                msg = "The number of sites does not match the number of coordinates."
                logger.error(msg)
                raise RuntimeError(msg)
            logger.info("The new universe has {1} beads in {0} frames.".format(
                *coordinates.shape))

            universe.load_new(coordinates, format=MemoryReader)
            universe.trajectory.dimensions_array = dimensions_array.copy()
        if universe.trajectory.ts.has_velocities:
            velocities: np.ndarray = np.concatenate(velocities, axis=1)
            universe.trajectory.velocity_array: np.ndarray = velocities.copy()
        if universe.trajectory.ts.has_forces:
            forces: np.ndarray = np.concatenate(forces, axis=1)
            universe.trajectory.force_array: np.ndarray = forces.copy()

    return universe


def rename_universe(universe: mda.Universe) -> mda.Universe:
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

    Returns
    -------
    :class:`~MDAnalysis.Universe`
        The universe with renamed residues and atoms.
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
    return universe
