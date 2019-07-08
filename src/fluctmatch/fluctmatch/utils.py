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
import logging
import shutil
import subprocess
import tempfile
import textwrap
from contextlib import ExitStack
from pathlib import Path

import MDAnalysis as mda
import MDAnalysis.analysis.base as analysis
import click
import numpy as np
import pandas as pd

from fluctmatch.fluctmatch.data import charmm_split

logger = logging.getLogger(__name__)


class AverageStructure(analysis.AnalysisBase):
    def __init__(self, atomgroup, **kwargs):
        """Calculate the average structure of a trajectory.

        Parameters
        ----------
        atomgroup : :class:`~MDAnalysis.Universe.AtomGroup`
            An AtomGroup
        start : int, optional
            start frame of analysis
        stop : int, optional
            stop frame of analysis
        step : int, optional
            number of frames to skip between each analysed frame
        verbose : bool, optional
            Turn on verbosity
        """
        super().__init__(atomgroup.universe.trajectory, **kwargs)
        self._ag = atomgroup
        self._nframes = atomgroup.universe.trajectory.n_frames

    def _prepare(self):
        self.result = np.zeros_like(self._ag.positions)

    def _single_frame(self):
        self.result += self._ag.positions

    def _conclude(self):
        self.result /= self._nframes


class BondAverage(analysis.AnalysisBase):
    def __init__(self, atomgroup, **kwargs):
        """Calculate the average bond length.

        Parameters
        ----------
        atomgroup : :class:`~MDAnalysis.Universe.AtomGroup`
            An AtomGroup
        start : int, optional
            start frame of analysis
        stop : int, optional
            stop frame of analysis
        step : int, optional
            number of frames to skip between each analysed frame
        verbose : bool, optional
            Turn on verbosity
        """
        super().__init__(atomgroup.universe.trajectory, **kwargs)
        self._ag = atomgroup
        self._nframes = atomgroup.universe.trajectory.n_frames

    def _prepare(self):
        self.result = np.zeros_like(self._ag.bonds.bonds())

    def _single_frame(self):
        self.result += self._ag.bonds.bonds()

    def _conclude(self):
        self.result = np.rec.fromarrays(
            [
                self._ag.bonds.atom1.names,
                self._ag.bonds.atom2.names,
                self.result / self._nframes,
            ],
            names=["I", "J", "r_IJ"],
        )
        self.result = pd.DataFrame.from_records(self.result)


class BondStd(analysis.AnalysisBase):
    def __init__(self, atomgroup, average, **kwargs):
        """Calculate the fluctuation in bond lengths.

        Parameters
        ----------
        atomgroup : :class:`~MDAnalysis.Universe.AtomGroup`
            An AtomGroup
        average : float or ""lass:`~numpy.array`
            Average bond length
        start : int, optional
            start frame of analysis
        stop : int, optional
            stop frame of analysis
        step : int, optional
            number of frames to skip between each analysed frame
        verbose : bool, optional
            Turn on verbosity
        """
        super().__init__(atomgroup.universe.trajectory, **kwargs)
        self._ag = atomgroup
        self._nframes = atomgroup.universe.trajectory.n_frames
        self._average = average

    def _prepare(self):
        self.result = np.zeros_like(self._ag.bonds.bonds())

    def _single_frame(self):
        self.result += np.square(self._ag.bonds.bonds() - self._average)

    def _conclude(self):
        self.result = np.rec.fromarrays(
            [
                self._ag.bonds.atom1.names,
                self._ag.bonds.atom2.names,
                np.sqrt(self.result / self._nframes),
            ],
            names=["I", "J", "r_IJ"],
        )
        self.result = pd.DataFrame.from_records(self.result)


def write_charmm_files(
    universe, outdir=Path.cwd(), prefix="cg", write_traj=True, **kwargs
):
    """Write CHARMM coordinate, topology PSF, stream, and topology RTF files.

    Parameters
    ----------
    universe : :class:`~MDAnalysis.Universe` or :class:`~MDAnalysis.AtomGroup`
        A collection of atoms in a universe or AtomGroup with bond definitions.
    outdir : str
        Location to write the files.
    prefix : str
        Prefix of filenames
    write_traj : bool
        Write the trajectory to disk.
    charmm_version
        Version of CHARMM for formatting (default: 41)
    extended
        Use the extended format.
    cmap
        Include CMAP section.
    cheq
        Include charge equilibration.
    title
        Title lines at the beginning of the file.
    """
    from MDAnalysis.core import topologyattrs

    # Attempt to create the necessary subdirectory
    Path(outdir).mkdir(parents=True, exist_ok=True)

    filename = Path(outdir) / prefix
    filenames = dict(
        psf_file=filename.with_suffix(".psf"),
        xplor_psf_file=filename.with_suffix(".xplor.psf"),
        crd_file=filename.with_suffix(".cor"),
        stream_file=filename.with_suffix(".stream"),
        topology_file=filename.with_suffix(".rtf"),
        traj_file=filename.with_suffix(".dcd"),
    )

    n_atoms = universe.atoms.n_atoms
    n_bonds = len(universe.bonds)
    n_angles = len(universe.angles)
    n_dihedrals = len(universe.dihedrals)
    n_impropers = len(universe.impropers)
    logger.warning(
        f"The system has {n_atoms:d} atoms, {n_bonds:d} bonds, "
        f"{n_angles:d} angles, {n_dihedrals:d} dihedrals, and "
        f"{n_impropers:d} impropers. Depending upon the size of "
        f"the system, file writing may take a while and have a "
        f"large file size."
    )

    # Write required CHARMM input files.
    with ExitStack() as stack:
        rtf = mda.Writer(filenames["topology_file"].as_posix(), **kwargs)
        stream = mda.Writer(filenames["stream_file"].as_posix(), **kwargs)
        psf = mda.Writer(filenames["psf_file"].as_posix(), **kwargs)
        xplor = mda.Writer(filenames["xplor_psf_file"].as_posix(), **kwargs)

        logger.info(f"Writing {rtf.filename}...")
        rtf.write(universe)

        logger.info(f"Writing {stream.filename}...")
        stream.write(universe)

        logger.info(f"Writing {psf.filename}...")
        psf.write(universe)

        # Write an XPLOR version of the PSF
        atomtypes = topologyattrs.Atomtypes(universe.atoms.names)
        universe._topology.add_TopologyAttr(topologyattr=atomtypes)
        universe._generate_from_topology()
        logger.info(f"Writing {xplor.filename}...")
        psf.write(universe)

    # Write the new trajectory in Gromacs XTC format.
    if write_traj:
        universe.trajectory.rewind()
        with ExitStack() as stack:
            trj = stack.enter_context(
                mda.Writer(
                    filenames["traj_file"].as_posix(),
                    universe.atoms.n_atoms,
                    istart=universe.trajectory.time,
                    remarks="Written by fluctmatch.",
                )
            )
            bar = stack.enter_context(click.progressbar(universe.trajectory))
            logger.info(f"Writing the trajectory {trj.filename}...")
            logger.warning(
                "This may take a while depending upon the size and "
                "length of the trajectory."
            )
            for ts in bar:
                trj.write(ts)

    # Calculate the average coordinates from the trajectory.
    logger.info("Determining the average structure of the trajectory. ")
    logger.warning(
        "Note: This could take a while depending upon the " "size of your trajectory."
    )
    positions = AverageStructure(universe.atoms).run().result
    positions = positions.reshape((*positions.shape, 1))

    # Create a new universe.
    topologies = ("names", "resids", "resnums", "resnames", "segids")
    avg_universe = mda.Universe.empty(
        n_atoms=n_atoms,
        n_residues=universe.residues.n_residues,
        n_segments=universe.segments.n_segments,
        atom_resindex=universe.atoms.resindices,
        residue_segindex=universe.residues.segindices,
        trajectory=True,
    )
    for _ in topologies:
        avg_universe.add_TopologyAttr(_)
    avg_universe.atoms.names = universe.atoms.names
    avg_universe.residues.resids = universe.residues.resids
    avg_universe.residues.resnums = universe.residues.resnums
    avg_universe.residues.resnames = universe.residues.resnames
    avg_universe.segments.segids = universe.segments.segids
    avg_universe.load_new(positions, order="acf")

    # avg_universe.load_new(
    #     positions, )
    with mda.Writer(filenames["crd_file"].as_posix(), dt=1.0, **kwargs) as crd:
        logger.info(f"Writing {crd.filename}...")
        crd.write(avg_universe.atoms)


def split_gmx(info, data_dir=Path.cwd() / "data", **kwargs):
    """Create a subtrajectory from a Gromacs trajectory.

    Parameters
    ----------
    info : :class:`collections.namedTuple`
        Contains information about the data subdirectory and start and
        stop frames
    data_dir : str, optional
        Location of the main data directory
    topology : str, optional
        Topology filename (e.g., tpr gro g96 pdb brk ent)
    trajectory : str, optional
        A Gromacs trajectory file (e.g., xtc trr)
    index : str, optional
        A Gromacs index file (e.g., ndx)
    outfile : str, optional
        A Gromacs trajectory file (e.g., xtc trr)
    logfile : str, optional
        Log file for output of command
    system : int
        Atom selection from Gromacs index file (0 = System, 1 = Protein)
    """
    # Trajectory splitting information
    subdir, start, stop = info
    subdir = Path(data_dir) / subdir
    gromacs_exec = shutil.which("gmx")

    # Attempt to create the necessary subdirectory
    subdir.mkdir(parents=True, exist_ok=True)

    # Various filenames
    topology = kwargs.get("topology", Path.cwd() / "md.tpr")
    trajectory = kwargs.get("trajectory", Path.cwd() / "md.xtc")
    index = kwargs.get("index")
    outfile = subdir / kwargs.get("outfile", Path.cwd() / "aa.xtc")
    logfile = subdir / kwargs.get("logfile", Path.cwd() / "split.log")

    if index is not None:
        command = [
            "gmx",
            "trjconv",
            "-s",
            topology,
            "-f",
            trajectory,
            "-n",
            index,
            "-o",
            outfile,
            "-b",
            f"{start:d}",
            "-e",
            f"{stop:d}",
        ]
    else:
        command = [
            gromacs_exec,
            "trjconv",
            "-s",
            topology,
            "-f",
            trajectory,
            "-o",
            outfile,
            "-b",
            f"{start:d}",
            "-e",
            f"{stop:d}",
        ]
    fd, fpath = tempfile.mkstemp(text=True)
    fpath = Path(fpath)
    with open(fpath, mode="w") as temp:
        with ExitStack() as stack:
            temp = open(fpath, mode="w+")
            log = open(logfile, mode="w")
            logger.info(
                f"Writing trajectory to {outfile} and "
                f"writing Gromacs output to {logfile}"
            )
            print(kwargs.get("system", 0), file=temp)
            temp.seek(0)
            subprocess.check_call(
                command, stdin=temp, stdout=log, stderr=subprocess.STDOUT
            )
    fpath.unlink()


def split_charmm(info, data_dir=Path.cwd() / "data", **kwargs):
    """Create a subtrajectory from a CHARMM trajectory.

    Parameters
    ----------
    info : :class:`collections.namedTuple`
        Contains information about the data subdirectory and start and
        stop frames
    data_dir : str, optional
        Location of the main data directory
    toppar : str, optional
        Directory containing CHARMM topology/parameter files
    trajectory : str, optional
        A CHARMM trajectory file (e.g., dcd)
    outfile : str, optional
        A CHARMM trajectory file (e.g., dcd)
    logfile : str, optional
        Log file for output of command
    charmm_version : int
        Version of CHARMM
    """
    # Trajectory splitting information
    subdir, start, stop = info
    subdir = Path(data_dir) / subdir
    charmm_exec = shutil.which("charmm")

    # Attempt to create the necessary subdirectory
    subdir.mkdir(parents=True, exist_ok=True)

    # Various filenames
    version = kwargs.get("charmm_version", 41)
    toppar = kwargs.get("toppar", f"/opt/local/charmm/c{version:d}b1/toppar")
    trajectory = kwargs.get("trajectory", Path.cwd() / "md.dcd")
    outfile = subdir / kwargs.get("outfile", Path.cwd() / "aa.dcd")
    logfile = subdir / kwargs.get("logfile", Path.cwd() / "split.log")
    inpfile = subdir / "split.inp"

    with open(inpfile, mode="w") as charmm_input:
        charmm_inp = charmm_split.split_inp.format(
            toppar=toppar,
            trajectory=trajectory,
            outfile=outfile,
            version=version,
            start=start,
            stop=stop,
        )
        charmm_inp = textwrap.dedent(charmm_inp[1:])
        print(charmm_inp, file=charmm_input)
    command = [charmm_exec, "-i", inpfile, "-o", subdir / logfile]
    subprocess.check_call(command)
