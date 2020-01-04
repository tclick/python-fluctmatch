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

import logging
import shutil
import subprocess
import tempfile
import textwrap
from contextlib import ExitStack
from pathlib import Path
from typing import Mapping

import click
import MDAnalysis as mda
import MDAnalysis.analysis.base as analysis
import numpy as np

from ..fluctmatch.data import charmm_split

logger: logging.Logger = logging.getLogger(__name__)


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


class BondStats(analysis.AnalysisBase):
    """Calculate the average and fluctuations of bond distances."""

    def __init__(self, atomgroup: mda.AtomGroup, **kwargs: Mapping):
        """
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

        self._ag: mda.AtomGroup = atomgroup
        self._nframes: int = atomgroup.universe.trajectory.n_frames
        self._nbonds: int = len(self._ag.universe.bonds)

    def _prepare(self):
        self.result: np.recarray = np.recarray((self._nbonds,),
                                               dtype=[("x", float),
                                                      ("x2", float)])
        self.result.x: np.ndarray = np.zeros(self._nbonds)
        self.result.x2: np.ndarray = np.zeros(self._nbonds)

    def _single_frame(self):
        self.result.x += self._ag.bonds.bonds()
        self.result.x2 += np.square(self._ag.bonds.bonds())

    def _conclude(self):
        results: np.recarray = np.recarray((self._nbonds,),
                                           dtype=[("average", float),
                                                  ("stddev", float)])
        results.average = self.result.x / self._nframes

        # Polynomial expansion of standard deviation.
        # sum(x - m)^2 = nx^2 - 2nxm + nm^2, where n is the number of frames.
        # The summation of x and x^2 occur in the _single_frame method, so the
        # equation can be reduced to sum(x - m)^2 = x^2 - 2xm + nm^2.
        results.stddev = (self.result.x2 - 2 * results.average * self.result.x
                          + self._nframes * np.square(results.average))
        results.stddev /= self._nframes
        results.stddev = np.sqrt(results.stddev)
        self.result = results.copy()


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
        rtf = stack.enter_context(
            mda.Writer(filenames["topology_file"].as_posix(), **kwargs))
        stream = stack.enter_context(
            mda.Writer(filenames["stream_file"].as_posix(), **kwargs))
        psf = stack.enter_context(
            mda.Writer(filenames["psf_file"].as_posix(), **kwargs))
        xplor = stack.enter_context(
            mda.Writer(filenames["xplor_psf_file"].as_posix(), **kwargs))

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
            trj = stack.enter_context(mda.Writer(
                filenames["traj_file"],
                universe.atoms.n_atoms,
                istart=universe.trajectory.time,
                remarks="Written by fluctmatch."))
            bar = stack.enter_context(click.progressbar(universe.trajectory))
            logger.info("Writing the trajectory {}...".format(
                filenames["traj_file"]))
            logger.warning("This may take a while depending upon the size and "
                           "length of the trajectory.")
            for ts in bar:
                trj.write(ts)

    # Write an XPLOR version of the PSF
    atomtypes = topologyattrs.Atomtypes(universe.atoms.names)
    universe._topology.add_TopologyAttr(topologyattr=atomtypes)
    universe._generate_from_topology()
    with mda.Writer(filenames["xplor_psf_file"], **kwargs) as psf:
        logger.info("Writing {}...".format(filenames["xplor_psf_file"]))
        psf.write(universe)

    # Calculate the average coordinates from the trajectory.
    logger.info("Determining the average structure of the trajectory. ")
    logger.warning("Note: This could take a while depending upon the size of "
                   "your trajectory.")
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
    with ExitStack() as stack:
        temp = stack.enter_context(open(fpath, mode="w+"))
        log = stack.enter_context(open(logfile, mode="w"))
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
