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
"""Fluctuation matching using CHARMM.

Notes
------
For CHARMM to work with the fluctuation matching code, it must be
recompiled with some modifications to the source code. `ATBMX`, `MAXATC`,
`MAXCB` (located in dimens.fcm [c35] or dimens_ltm.src [c39]) must
be increased. `ATBMX` determines the number of bonds allowed per
atom, `MAXATC` describes the maximum number of atom types, and `MAXCB`
determines the maximum number of bond parameters in the CHARMM parameter
file. Additionally, `CHSIZE` may need to be increased if using an earlier
version (< c36).
"""

import copy
import logging
import os
import shutil
import subprocess
import textwrap
import time
from contextlib import ExitStack

import MDAnalysis as mda
import numpy as np
import pandas as pd
from MDAnalysis.coordinates.core import reader
from scipy import constants

from fluctmatch.fluctmatch import base as fmbase
from fluctmatch.fluctmatch.data import charmm_init
from fluctmatch.fluctmatch.data import charmm_nma
from fluctmatch.fluctmatch.data import charmm_thermo
from fluctmatch.intcor import utils as icutils
from fluctmatch.parameter import utils as prmutils

logger = logging.getLogger(__name__)


class CharmmFluctMatch(fmbase.FluctMatch):
    """Fluctuation matching using CHARMM."""
    bond_def = ["I", "J"]
    error_hdr = ["step", "Kb_rms", "fluct_rms", "b0_rms"]

    def __init__(self, *args, **kwargs):
        """Initialization of fluctuation matching using the CHARMM program.

        Parameters
        ----------
        topology : filename or Topology object
            A CHARMM/XPLOR PSF topology file, PDB file or Gromacs GRO file;
            used to define the list of atoms. If the file includes bond
            information, partial charges, atom masses, ... then these data will
            be available to MDAnalysis. A "structure" file (PSF, PDB or GRO, in
            the sense of a topology) is always required. Alternatively, an
            existing :class:`MDAnalysis.core.topology.Topology` instance may
            also be given.
        topology_format
            Provide the file format of the topology file; ``None`` guesses it
            from the file extension [``None``] Can also pass a subclass of
            :class:`MDAnalysis.topology.base.TopologyReaderBase` to define a
            custom reader to be used on the topology file.
        format
            Provide the file format of the coordinate or trajectory file;
            ``None`` guesses it from the file extension. Note that this keyword
            has no effect if a list of file names is supplied because the
            "chained" reader has to guess the file format for each individual
            list member. [``None``] Can also pass a subclass of
            :class:`MDAnalysis.coordinates.base.ProtoReader` to define a custom
            reader to be used on the trajectory file.
        guess_bonds : bool, optional
            Once Universe has been loaded, attempt to guess the connectivity
            between atoms.  This will populate the .bonds .angles and .dihedrals
            attributes of the Universe.
        vdwradii : dict, optional
            For use with *guess_bonds*. Supply a dict giving a vdwradii for each
            atom type which are used in guessing bonds.
        is_anchor : bool, optional
            When unpickling instances of
            :class:`MDAnalysis.core.groups.AtomGroup` existing Universes are
            searched for one where to anchor those atoms. Set to ``False`` to
            prevent this Universe from being considered. [``True``]
        anchor_name : str, optional
            Setting to other than ``None`` will cause
            :class:`MDAnalysis.core.groups.AtomGroup` instances pickled from
            the Universe to only unpickle if a compatible Universe with matching
            *anchor_name* is found. Even if *anchor_name* is set *is_anchor*
            will still be honored when unpickling.
        in_memory
            After reading in the trajectory, transfer it to an in-memory
            representations, which allow for manipulation of coordinates.
        in_memory_step
            Only read every nth frame into in-memory representation.
        outdir
            Output directory
        temperature
            Temperature (in K)
        rmin
            Minimum distance to consider for bond lengths.
        rmax
            Maximum distance to consider for bond lengths.
        charmm_version
            Version of CHARMM for formatting (default: 41)
        extended
            Use the extended format.
        title
            Title lines at the beginning of the file.
        resid
            Include segment IDs in the internal coordinate files.
        nonbonded
            Include the nonbonded section in the parameter file.
        """
        super().__init__(*args, **kwargs)
        self.dynamic_params = dict()
        self.filenames = dict(
            init_input=self.outdir / "fluctinit.inp",
            init_log=self.outdir / "fluctinit.log",
            init_avg_ic=self.outdir / "init.average.ic",
            init_fluct_ic=self.outdir / "init.fluct.ic",
            avg_ic=self.outdir / "average.ic",
            fluct_ic=self.outdir / "fluct.ic",
            dynamic_prm=self.outdir / self.prefix.with_prefix(".dist.prm"),
            fixed_prm=self.outdir / self.prefix.with_prefix(".prm"),
            psf_file=self.outdir / self.prefix.with_prefix(".psf"),
            xplor_psf_file=self.outdir / self.prefix.with_prefix(".xplor.psf"),
            crd_file=self.outdir / self.prefix.with_prefix(".cor"),
            stream_file=self.outdir / self.prefix.with_prefix(".stream"),
            topology_file=self.outdir / self.prefix.with_prefix(".rtf"),
            nma_crd=self.outdir / self.prefix.with_prefix(".mini.cor"),
            nma_vib=self.outdir / self.prefix.with_prefix(".vib"),
            charmm_input=self.outdir / self.prefix.with_prefix(".inp"),
            charmm_log=self.outdir / self.prefix.with_prefix(".log"),
            error_data=self.outdir / "error.dat",
            thermo_input=self.outdir / "thermo.inp",
            thermo_log=self.outdir / "thermo.log",
            thermo_data=self.outdir / "thermo.dat",
            traj_file=(
                self.args[1] if len(self.args) > 1 else self.outdir / "cg.dcd")
        )

        # Location of CHARMM executable
        self.charmmexec = os.environ.get("CHARMMEXEC", shutil.which("charmm"))

        # Boltzmann constant
        self.BOLTZ = self.temperature * (constants.k * constants.N_A
                                         / (constants.calorie * constants.kilo))

        # Bond factor mol^2-Ang./kcal^2
        self.KFACTOR = 0.02

        # Self consistent error information.
        self.error = pd.DataFrame(
            np.zeros((1, len(self.error_hdr)), dtype=np.int),
            columns=self.error_hdr,
        )

    def _create_ic_table(self, universe, data):
        data.set_index(self.bond_def, inplace=True)
        table = icutils.create_empty_table(universe.atoms)
        hdr = table.columns
        table.set_index(self.bond_def, inplace=True)
        table.drop(["r_IJ", ], axis=1, inplace=True)
        table = pd.concat([table, data["r_IJ"]], axis=1)
        return table.reset_index()[hdr]

    def initialize(self, nma_exec=None, restart=False):
        """Create an elastic network model from a basic coarse-grain model.

        Parameters
        ----------
        nma_exec : str
            executable file for normal mode analysis
        restart : bool, optional
            Reinitialize the object by reading files instead of doing initial
            calculations.
        """
        if not restart:
            # Write CHARMM input file.
            if not self.filenames["init_input"].exists():
                version = self.kwargs.get("charmm_version", 41)
                dimension = (
                    "dimension chsize 1000000" if version >= 36 else "")
                with open(self.filenames["init_input"], "w") as charmm_file:
                    logger.info("Writing CHARMM input file.")
                    charmm_inp = charmm_init.init.format(
                        flex="flex" if version else "",
                        version=version,
                        dimension=dimension,
                        **self.filenames)
                    charmm_inp = textwrap.dedent(charmm_inp.strip("\n"))
                    charmm_file.write(charmm_inp)

            charmm_exec = self.charmmexec if nma_exec is None else nma_exec
            with ExitStack() as stack:
                log_file = stack.enter_context(
                    open(self.filenames["init_log"], "w"))
                std_ic = stack.enter_context(
                    reader(self.filenames["init_fluct_ic"]))
                avg_ic = stack.enter_context(
                    reader(self.filenames["init_avg_ic"]))
                fixed = stack.enter_context(
                    mda.Writer(self.filenames["fixed_prm"], **self.kwargs))
                dynamic = stack.enter_context(
                    mda.Writer(self.filenames["dynamic_prm"], **self.kwargs))

                subprocess.check_call(
                    [charmm_exec, "-i", self.filenames["init_input"]],
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                )

                # Write the parameter files.
                std_bonds = std_ic.read().set_index(self.bond_def)
                avg_bonds = avg_ic.read().set_index(self.bond_def)
                target = pd.concat([std_bonds["r_IJ"], avg_bonds["r_IJ"]],
                                   axis=1)
                target.reset_index(inplace=True)

                logger.info("Calculating the initial CHARMM parameters...")
                universe = mda.Universe(
                    self.filenames["xplor_psf_file"], self.filenames["crd_file"]
                )
                self.target = prmutils.create_empty_parameters(universe,
                                                               **self.kwargs)
                target.columns = self.target["BONDS"].columns
                self.target["BONDS"] = target.copy(deep=True)
                self.parameters = copy.deepcopy(self.target)
                self.parameters["BONDS"]["Kb"] = (
                    self.BOLTZ / np.square(self.parameters["BONDS"]["Kb"]))
                self.dynamic_params = copy.deepcopy(self.parameters)
                logger.info(f"Writing {self.filenames['fixed_prm']}...")
                fixed.write(self.parameters)
                logger.info(f"Writing {self.filenames['dynamic_prm']}...")
                dynamic.write(self.dynamic_params)
        else:
            if not self.filenames["fixed_prm"].exists():
                self.initialize(nma_exec)
            try:
                # Read the parameter files.
                logger.info("Loading parameter and internal coordinate files.")
                with ExitStack() as stack:
                    fixed = stack.enter_context(
                        reader(self.filenames["fixed_prm"]))
                    dynamic = stack.enter_context(
                        reader(self.filenames["dynamic_prm"]))
                    init_avg = stack.enter_context(
                        reader(self.filenames["init_avg_ic"]))
                    init_fluct = stack.enter_context(
                        reader(self.filenames["init_fluct_ic"]))

                    self.parameters.update(fixed.read())
                    self.dynamic_params.update(dynamic.read())

                    # Read the initial internal coordinate files.
                    avg_table = init_avg.read().set_index(
                        self.bond_def)["r_IJ"]
                    fluct_table = (init_fluct.read().set_index(
                        self.bond_def)["r_IJ"])
                    table = pd.concat([fluct_table, avg_table], axis=1)

                    # Set the target fluctuation values.
                    logger.info("Files loaded successfully...")
                    self.target = copy.deepcopy(self.parameters)
                    self.target["BONDS"].set_index(self.bond_def, inplace=True)
                    table.columns = self.target["BONDS"].columns
                    self.target["BONDS"] = table.copy(deep=True).reset_index()
            except (FileNotFoundError, IOError):
                raise IOError("Some files are missing. Unable to restart.")

    def run(self, nma_exec=None, tol=1.e-4, n_cycles=250):
        """Perform a self-consistent fluctuation matching.

        Parameters
        ----------
        nma_exec : str
            executable file for normal mode analysis
        tol : float, optional
            error tolerance
        n_cycles : int, optional
            number of fluctuation matching cycles
        """
        # Find CHARMM executable
        charmm_exec = self.charmmexec if nma_exec is None else nma_exec
        if charmm_exec is None:
            error_msg = ("Please set CHARMMEXEC with the location of your "
                         "CHARMM executable file or add the charmm path to "
                         "your PATH environment.")
            logger.exception(error_msg)
            OSError(error_msg)

        # Read the parameters
        if not self.parameters:
            try:
                self.initialize(nma_exec, restart=True)
            except IOError:
                IOError("Some files are missing. Unable to restart.")

        # Write CHARMM input file.
        if not self.filenames["charmm_input"].exists():
            version = self.kwargs.get("charmm_version", 41)
            dimension = ("dimension chsize 1000000" if version >= 36 else "")
            with open(self.filenames["charmm_input"], "w") as charmm_file:
                logger.info("Writing CHARMM input file.")
                charmm_inp = charmm_nma.nma.format(
                    temperature=self.temperature,
                    flex="flex" if version else "",
                    version=version,
                    dimension=dimension,
                    **self.filenames)
                charmm_inp = textwrap.dedent(charmm_inp.strip("\n"))
                charmm_file.write(charmm_inp)

        # Set the indices for the parameter tables.
        self.target["BONDS"].set_index(self.bond_def, inplace=True)
        bond_values = self.target["BONDS"].columns

        # Check for restart.
        try:
            if self.filenames["error_data"].stat().st_size > 0:
                with open(self.filenames["error_data"]) as data:
                    error_info = pd.read_csv(
                        data,
                        header=0,
                        skipinitialspace=True,
                        delim_whitespace=True)
                    if not error_info.empty:
                        self.error["step"] = error_info["step"].values[-1]
            else:
                raise FileNotFoundError
        except (FileNotFoundError, OSError):
            with open(self.filenames["error_data"], "w") as data:
                np.savetxt(data, [self.error_hdr, ], fmt="%10s", delimiter="")
        self.error["step"] += 1

        # Run simulation
        logger.info("Starting fluctuation matching")
        st = time.time()

        for i in range(n_cycles):
            self.error["step"] = i + 1
            with ExitStack() as stack:
                log_file = stack.enter_context(
                    open(self.filenames["charmm_log"], "w"))
                subprocess.check_call(
                    [charmm_exec, "-i", self.filenames["charmm_input"]],
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                )
                self.dynamic_params["BONDS"].set_index(self.bond_def,
                                                       inplace=True)
                self.parameters["BONDS"].set_index(self.bond_def, inplace=True)

                # Read the average bond distance.
                avg_intcor = stack.enter_context(
                    reader(self.filenames["avg_ic"]))
                fluct_intcor = stack.enter_context(
                    reader(self.filenames["avg_ic"]))
                avg_ic = avg_intcor.read().set_index(self.bond_def)["r_IJ"]
                fluct_ic = fluct_intcor.read().set_index(self.bond_def)["r_IJ"]

            vib_ic = pd.concat([fluct_ic, avg_ic], axis=1)
            vib_ic.columns = bond_values

            # Calculate the r.m.s.d. between fluctuation and distances
            # compared with the target values.
            vib_error = self.target["BONDS"] - vib_ic
            vib_error = vib_error.apply(np.square).mean(axis=0)
            vib_error = np.sqrt(vib_error)
            self.error[self.error.columns[-2:]] = vib_error.T.values

            # Calculate the new force constant.
            optimized = vib_ic.apply(np.reciprocal).apply(np.square)
            target = self.target["BONDS"].apply(np.reciprocal).apply(np.square)
            optimized -= target
            optimized *= self.BOLTZ * self.KFACTOR
            vib_ic[bond_values[0]] = (self.parameters["BONDS"][bond_values[0]]
                                      - optimized[bond_values[0]])
            vib_ic[bond_values[0]] = vib_ic[bond_values[0]].apply(
                lambda x: np.clip(x, a_min=0, a_max=None))

            # r.m.s.d. between previous and current force constant
            diff = self.dynamic_params["BONDS"] - vib_ic
            diff = diff.apply(np.square).mean(axis=0)
            diff = np.sqrt(diff)
            self.error[self.error.columns[1]] = diff.values[0]

            # Update the parameters and write to file.
            self.parameters["BONDS"][bond_values[0]] = (
                vib_ic[bond_values[0]].copy(deep=True))
            self.dynamic_params["BONDS"] = vib_ic.copy(deep=True)
            self.parameters["BONDS"].reset_index(inplace=True)
            self.dynamic_params["BONDS"].reset_index(inplace=True)

            with ExitStack() as stack:
                fixed_prm = stack.enter_context(
                    mda.Writer(self.filenames["fixed_prm"], **self.kwargs))
                dynamic_prm = stack.enter_context(
                    mda.Writer(self.filenames["dynamic_prm"], **self.kwargs))
                error_file = stack.enter_context(
                    open(self.filenames["error_data"], "a"))

                fixed_prm.write(self.parameters)
                dynamic_prm.write(self.dynamic_params)
                np.savetxt(
                    error_file,
                    self.error,
                    fmt="%10d%10.6f%10.6f%10.6f",
                    delimiter="",
                )

            if (self.error[self.error.columns[1]] < tol).bool():
                break

        logger.info(f"Fluctuation matching completed in {time.time() - st:.6f}")
        self.target["BONDS"] = self.target["BONDS"].reset_index()

    def calculate_thermo(self, nma_exec=None):
        """Calculate the thermodynamic properties of the trajectory.

        Parameters
        ----------
        nma_exec : str
            executable file for normal mode analysis
        """
        # Find CHARMM executable
        charmm_exec = self.charmmexec if nma_exec is None else nma_exec
        if charmm_exec is None:
            error_msg = (
                "Please set CHARMMEXEC with the location of your CHARMM "
                "executable file or add the charmm path to your PATH "
                "environment.")
            logger.exception(error_msg)
            OSError(error_msg)

        if not self.filenames["thermo_input"].exists():
            version = self.kwargs.get("charmm_version", 41)
            dimension = ("dimension chsize 500000 maxres 3000000"
                         if version >= 36 else "")
            with open(self.filenames["thermo_input"], "w") as charmm_file:
                logger.info("Writing CHARMM input file.")
                charmm_inp = charmm_thermo.thermodynamics.format(
                    trajectory=self.outdir / self.args[-1],
                    temperature=self.temperature,
                    flex="flex" if version else "",
                    version=version,
                    dimension=dimension,
                    **self.filenames)
                charmm_inp = textwrap.dedent(charmm_inp.strip("\n"))
                charmm_file.write(charmm_inp)

        # Calculate thermodynamic properties of the trajectory.
        with open(self.filenames["thermo_log"], "w") as log_file:
            logger.info("Running thermodynamic calculation.")
            subprocess.check_call(
                [charmm_exec, "-i", self.filenames["thermo_input"]],
                stdout=log_file,
                stderr=subprocess.STDOUT,
            )
            logger.info("Calculations completed.")

        # Read log file
        with ExitStack as stack:
            log_file = stack.enter_context(open(self.filenames["thermo_log"]))
            data_file = stack.enter_context(
                open(self.filenames["thermo_data"], "w"))

            logger.info("Reading CHARMM log file.")
            header = ("SEGI  RESN  RESI     Entropy    Enthalpy     "
                      "Heatcap     Atm/res   Ign.frq")
            columns = np.array(header.split())
            columns[:3] = np.array(["segidI", "RESN", "resI"])
            thermo = []
            for line in log_file:
                if line.find(header) < 0:
                    continue
                if line.strip():
                    break
                thermo.append(line.strip().split())

            # Create human-readable table
            logger.info("Writing thermodynamics data file.")
            (pd.DataFrame(thermo, columns=columns)
             .drop(["RESN", "Atm/res", "Ign.frq"], axis=1)
             .to_csv(data_file, index=False, float_format="%.4f",
                     encoding="utf-8"))
