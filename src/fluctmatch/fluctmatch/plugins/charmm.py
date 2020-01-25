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
"""Fluctuation matching using CHARMM.

Notes
------
For CHARMM to work with the fluctuation matching code, it must be
recompiled with some modifications to the source code. `ATBMX`, `MAXATC`,
`MAXCB` (located in dimens.fcm [c35] or dimens_ltm.src [c39]) must
be increased. `ATBMX` determines the number of bonds allowed per
atom, `MAXATC` describes the maximum number of atom core, and `MAXCB`
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
from pathlib import Path
from typing import ClassVar
from typing import Dict
from typing import List
from typing import NoReturn
from typing import TextIO

import MDAnalysis as mda
import numpy as np
import pandas as pd
from MDAnalysis.coordinates.core import reader
from scipy import constants
from sklearn.metrics import mean_squared_error

from ...libs import intcor
from ...libs import parameters
from ..base import FluctMatchBase
from ..data import charmm_init
from ..data import charmm_nma
from ..data import charmm_thermo

logger: logging.Logger = logging.getLogger(__name__)


class FluctMatch(FluctMatchBase):
    """Fluctuation matching using CHARMM."""

    bond_def: ClassVar[List[str]] = ["I", "J"]
    error_hdr: ClassVar[List[str]] = ["Kb_rms", "fluct_rms", "b0_rms"]
    description: ClassVar[str] = "Fluctuation matching using CHARMM"

    def __init__(self, *args: List, **kwargs: Dict):
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
        self.dynamic_params: Dict[str, pd.DataFrame] = dict()
        self.filenames: Dict[str, Path] = dict(
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
                self.args[1] if len(self.args) > 1 else self.outdir / "cg.dcd"
            ),
        )

        # Location of CHARMM executable
        self.charmmexec: str = os.environ.get(
            "CHARMMEXEC", shutil.which("charmm")
        )

        # Boltzmann constant
        self.BOLTZ: float = self.temperature * (
            constants.k * constants.N_A / (constants.calorie * constants.kilo)
        )

        # Bond factor mol^2-Ang./kcal^2
        self.KFACTOR: float = 0.02

        # Self consistent error information.
        self.error: pd.DataFrame = pd.DataFrame(
            np.zeros_like(self.error_hdr, dtype=float), index=self.error_hdr).T
        self.error.index.name : str = "step"

    def _create_ic_table(
        self, universe: mda.Universe, data: pd.DataFrame
    ) -> pd.DataFrame:
        data: pd.DataFrame = data.set_index(self.bond_def)
        table = intcor.create_empty_table(universe.atoms)
        hdr: pd.DataFrame = table.columns
        table: pd.DataFrame = table.set_index(self.bond_def).drop(
            ["r_IJ"], axis=1
        )
        table: pd.DataFrame = pd.concat([table, data["r_IJ"]], axis=1)
        return table.reset_index()[hdr]

    def initialize(self, nma_exec: str = None, restart: bool = False) -> NoReturn:
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
                version: int = self.kwargs.get("charmm_version", 41)
                dimension: str = (
                    "dimension chsize 1000000" if version >= 36 else ""
                )
                with open(self.filenames["init_input"], "w") as charmm_file:
                    logger.info("Writing CHARMM input file.")
                    charmm_inp: str = charmm_init.init.format(
                        flex="flex" if version else "",
                        version=version,
                        dimension=dimension,
                        **self.filenames,
                    )
                    charmm_inp: str = textwrap.dedent(charmm_inp.strip("\n"))
                    charmm_file.write(charmm_inp)

            charmm_exec: str = self.charmmexec if nma_exec is None else nma_exec
            if charmm_exec is None:
                error_msg: str = (
                    "Please set CHARMMEXEC with the location of your "
                    "CHARMM executable file or add the charmm path to "
                    "your PATH environment."
                )
                logger.exception(error_msg)
                OSError(error_msg)
            with ExitStack() as stack:
                log_file: TextIO = stack.enter_context(
                    open(self.filenames["init_log"], "w")
                )
                std_ic: TextIO = stack.enter_context(
                    reader(self.filenames["init_fluct_ic"])
                )
                avg_ic: TextIO = stack.enter_context(
                    reader(self.filenames["init_avg_ic"])
                )
                fixed: TextIO = stack.enter_context(
                    mda.Writer(self.filenames["fixed_prm"], **self.kwargs)
                )
                dynamic: TextIO = stack.enter_context(
                    mda.Writer(self.filenames["dynamic_prm"], **self.kwargs)
                )

                subprocess.check_call(
                    [charmm_exec, "-i", self.filenames["init_input"]],
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                )

                # Write the parameter files.
                std_bonds: pd.DataFrame = std_ic.read().set_index(self.bond_def)
                avg_bonds: pd.DataFrame = avg_ic.read().set_index(self.bond_def)
                target: pd.DataFrame = pd.concat(
                    [std_bonds["r_IJ"], avg_bonds["r_IJ"]], axis=1
                ).reset_index()

                logger.info("Calculating the initial CHARMM parameters...")
                universe: mda.Universe = mda.Universe(
                    self.filenames["xplor_psf_file"], self.filenames["crd_file"]
                )
                self.target: Dict = parameters.create_empty_parameters(
                    universe, **self.kwargs
                )
                target.columns = self.target["BONDS"].columns
                self.target["BONDS"]: pd.DataFrame = target.copy(deep=True)
                self.parameters: Dict = copy.deepcopy(self.target)
                self.parameters["BONDS"]["Kb"]: pd.Series = (
                    self.BOLTZ / np.square(self.parameters["BONDS"]["Kb"])
                )
                self.dynamic_params: pd.DataFrame = copy.deepcopy(self.parameters)
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
                    fixed: TextIO = stack.enter_context(
                        reader(self.filenames["fixed_prm"])
                    )
                    dynamic: TextIO = stack.enter_context(
                        reader(self.filenames["dynamic_prm"])
                    )
                    init_avg: TextIO = stack.enter_context(
                        reader(self.filenames["init_avg_ic"])
                    )
                    init_fluct: TextIO = stack.enter_context(
                        reader(self.filenames["init_fluct_ic"])
                    )

                    self.parameters.update(fixed.read())
                    self.dynamic_params.update(dynamic.read())

                    # Read the initial internal coordinate files.
                    avg_table: pd.DataFrame = init_avg.read().set_index(
                        self.bond_def
                    )["r_IJ"]
                    fluct_table: pd.DataFrame = (
                        init_fluct.read().set_index(self.bond_def)["r_IJ"]
                    )
                    table: pd.DataFrame = pd.concat(
                        [fluct_table, avg_table], axis=1
                    )

                    # Set the target fluctuation values.
                    logger.info("Files loaded successfully...")
                    self.target: pd.DataFrame = copy.deepcopy(self.parameters)
                    self.target["BONDS"]: pd.DataFrame = self.target[
                        "BONDS"
                    ].set_index(self.bond_def)
                    table.columns = self.target["BONDS"].columns
                    self.target["BONDS"]: pd.Series = table.copy(
                        deep=True
                    ).reset_index()
            except (FileNotFoundError, IOError):
                raise IOError("Some files are missing. Unable to restart.")

    def run(
        self,
        nma_exec: str = None,
        tol: float = 1.0e-4,
        min_cycles: int = 200,
        max_cycles: int = 200,
        force_tol: float = 0.02,
    ) -> NoReturn:
        """Perform a self-consistent fluctuation matching.

        Parameters
        ----------
        nma_exec : str
            executable file for normal mode analysis
        tol : float, optional
            error tolerance
        min_cycles : int, optional
            minimum number of fluctuation matching cycles
        max_cycles : int, optional
            maximum number of fluctuation matching cycles
        force_tol : float, optional
            force constants <= force tolerance become zero after min_cycles
        """
        # Find CHARMM executable
        charmm_exec: str = self.charmmexec if nma_exec is None else nma_exec
        if charmm_exec is None:
            error_msg: str = (
                "Please set CHARMMEXEC with the location of your "
                "CHARMM executable file or add the charmm path to "
                "your PATH environment."
            )
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
            version: int = self.kwargs.get("charmm_version", 41)
            dimension: str = ("dimension chsize 1000000" if version >= 36 else "")
            with open(self.filenames["charmm_input"], "w") as charmm_file:
                logger.info("Writing CHARMM input file.")
                charmm_inp: str = charmm_nma.nma.format(
                    temperature=self.temperature,
                    flex="flex" if version else "",
                    version=version,
                    dimension=dimension,
                    **self.filenames,
                )
                charmm_inp: str = textwrap.dedent(charmm_inp.strip("\n"))
                charmm_file.write(charmm_inp)

        # Set the indices for the parameter tables.
        self.target["BONDS"]: pd.DataFrame = self.target["BONDS"].set_index(
            self.bond_def
        )
        bond_values: pd.DataFrame = self.target["BONDS"].columns

        # Check for restart.
        try:
            if self.filenames["error_data"].stat().st_size > 0:
                with open(self.filenames["error_data"]) as data:
                    error_info: pd.DataFrame = pd.read_csv(
                        data,
                        header=0,
                        skipinitialspace=True,
                        delim_whitespace=True,
                    )
                    if not error_info.empty:
                        self.error["step"] = error_info["step"].values[-1]
            else:
                raise FileNotFoundError
        except (FileNotFoundError, OSError):
            with open(self.filenames["error_data"], "w") as data:
                np.savetxt(data, [self.error_hdr], fmt="%10s", delimiter="")
        self.error["step"] += 1

        # Run simulation
        logger.info("Starting fluctuation matching")
        st: float = time.time()

        for i in range(1, max_cycles+1):
            with ExitStack() as stack:
                log_file: TextIO = stack.enter_context(
                    open(self.filenames["charmm_log"], "w")
                )
                subprocess.check_call(
                    [charmm_exec, "-i", self.filenames["charmm_input"]],
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                )
                self.dynamic_params["BONDS"]: pd.DataFrame = self.dynamic_params[
                    "BONDS"
                ].set_index(self.bond_def)
                self.parameters["BONDS"]: pd.DataFrame = self.parameters[
                    "BONDS"
                ].set_index(self.bond_def)

                # Read the average bond distance.
                avg_intcor: TextIO = stack.enter_context(
                    reader(self.filenames["avg_ic"])
                )
                fluct_intcor: TextIO = stack.enter_context(
                    reader(self.filenames["avg_ic"])
                )
                avg_ic: TextIO = avg_intcor.read().set_index(self.bond_def)[
                    "r_IJ"
                ]
                fluct_ic: TextIO = fluct_intcor.read().set_index(self.bond_def)[
                    "r_IJ"
                ]

            vib_ic: pd.DataFrame = pd.concat([fluct_ic, avg_ic], axis=1)
            vib_ic.columns = bond_values

            # Calculate the r.m.s.d. between fluctuation and distances
            # compared with the target values.
            vib_error: np.ndarray = mean_squared_error(self.target["BONDS"],
                                                       vib_ic,
                                                       multioutput="raw_output")
            self.error[self.error.columns[1:]] = np.sqrt(vib_error)

            # Calculate the new force constant.
            column = bond_values[0]
            optimized: pd.DataFrame = vib_ic.pow(-2)
            target: pd.DataFrame = self.target["BONDS"].pow(-2)
            optimized -= target
            optimized *= self.BOLTZ * self.KFACTOR
            vib_ic[column]: pd.DataFrame = self.parameters["BONDS"].sub(optimized)
            vib_ic[column]: pd.Series = vib_ic[column].apply(
                lambda x: np.clip(x, a_min=0, a_max=None)
            )
            if i > min_cycles:
                vib_ic[column][vib_ic[column] <= force_tol] = 0.0

            # r.m.s.d. between previous and current force constant
            diff: pd.Series = mean_squared_error(self.dynamic_params["BONDS"],
                                                 vib_ic,
                                                 multioutput="raw_values")
            self.error[self.error.columns[0]]: pd.Series = np.sqrt(diff.values[0])

            # Update the parameters and write to file.
            self.parameters["BONDS"][column]: pd.Series = (
                vib_ic[column].copy(deep=True)
            )
            self.dynamic_params["BONDS"]: pd.DataFrame = (
                vib_ic.copy(deep=True).reset_index()
            )
            self.parameters["BONDS"]: pd.DataFrame = (
                self.parameters["BONDS"].reset_index()
            )

            with ExitStack() as stack:
                fixed_prm: TextIO = stack.enter_context(
                    mda.Writer(self.filenames["fixed_prm"], **self.kwargs)
                )
                dynamic_prm: TextIO = stack.enter_context(
                    mda.Writer(self.filenames["dynamic_prm"], **self.kwargs)
                )
                error_file: TextIO = stack.enter_context(
                    open(self.filenames["error_data"], "a")
                )

                fixed_prm.write(self.parameters)
                dynamic_prm.write(self.dynamic_params)
                np.savetxt(
                    error_file,
                    self.error.reset_index(),
                    fmt="%10d%10.6f%10.6f%10.6f",
                    delimiter="",
                )

            if (self.error[self.error.columns[1]] < tol).bool():
                break

        logger.info(f"Fluctuation matching completed in {time.time() - st:.6f}")
        self.target["BONDS"] = self.target["BONDS"].reset_index()

    def calculate_thermo(self, nma_exec: str = None) -> NoReturn:
        """Calculate the thermodynamic properties of the trajectory.

        Parameters
        ----------
        nma_exec : str
            executable file for normal mode analysis
        """
        # Find CHARMM executable
        charmm_exec: str = self.charmmexec if nma_exec is None else nma_exec
        if charmm_exec is None:
            error_msg: str = (
                "Please set CHARMMEXEC with the location of your CHARMM "
                "executable file or add the charmm path to your PATH "
                "environment."
            )
            logger.exception(error_msg)
            OSError(error_msg)

        if not self.filenames["thermo_input"].exists():
            version: int = self.kwargs.get("charmm_version", 41)
            dimension: str = (
                "dimension chsize 500000 maxres 3000000" if version >= 36 else ""
            )
            with open(self.filenames["thermo_input"], "w") as charmm_file:
                logger.info("Writing CHARMM input file.")
                charmm_inp: str = charmm_thermo.thermodynamics.format(
                    trajectory=self.outdir / self.args[-1],
                    temperature=self.temperature,
                    flex="flex" if version else "",
                    version=version,
                    dimension=dimension,
                    **self.filenames,
                )
                charmm_inp: str = textwrap.dedent(charmm_inp.strip("\n"))
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
            log_file: TextIO = stack.enter_context(
                open(self.filenames["thermo_log"])
            )
            data_file: TextIO = stack.enter_context(
                open(self.filenames["thermo_data"], "w")
            )

            logger.info("Reading CHARMM log file.")
            header: str = (
                "SEGI  RESN  RESI     Entropy    Enthalpy     "
                "Heatcap     Atm/res   Ign.frq"
            )
            columns: np.ndarray = np.array(header.split())
            columns[:3] = np.array(["segidI", "RESN", "resI"])
            thermo: List = []
            for line in log_file:
                if line.find(header) < 0:
                    continue
                if line.strip():
                    break
                thermo.append(line.strip().split())

            # Create human-readable table
            logger.info("Writing thermodynamics data file.")
            (
                pd.DataFrame(thermo, columns=columns)
                .drop(["RESN", "Atm/res", "Ign.frq"], axis=1)
                .to_csv(
                    data_file, index=False, float_format="%.4f", encoding="utf-8"
                )
            )
