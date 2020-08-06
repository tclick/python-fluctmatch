# fluctmatch --- https://github.com/tclick/python-fluctmatch
# Copyright (c) 2013-2020 The fluctmatch Development Team and contributors
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
"""Class definition to calculate the initial fluctuations using CHARMM."""
import logging
import shutil
import subprocess
import textwrap
from pathlib import Path
from typing import NoReturn, Union

import numpy as np
import static_frame as sf
from jinja2 import Environment, PackageLoader, Template

from .. import base
from ...parsers.readers import IC

logger: logging.Logger = logging.getLogger(__name__)


class FluctMatch(base.FluctMatchBase):
    def __init__(
        self,
        *,
        temperature: float = 300.0,
        output_dir: Union[Path, str] = Path.home(),
        logfile: Union[Path, str] = "output.log",
        prefix: Union[Path, str] = "fluctmatch",
    ):
        """Initialization of fluctuation matching using CHARMM.

        Parameters
        ----------
        output_dir, Path or str
            Output directory
        temperature : float
            Temperature (in K)
        logfile : Path or str
            Output log file
        prefix : Path or str
            Filename prefix
        """
        super(FluctMatch, self).__init__(
            output_dir=output_dir,
            temperature=temperature,
            logfile=logfile,
            prefix=prefix,
        )
        title: str = """
            * Calculate the bond distance average and fluctuations.
            *
        """
        self.data.update(
            dict(
                title=textwrap.dedent(title[1:]),
                initialize=True,
                average=self.output_dir / "init.average.ic",
                fluctuation=self.output_dir / "init.fluct.ic",
            )
        )

    def calculate(self) -> sf.Frame:
        """Calculate the initial parameters from CHARMM fluctuations."""
        with IC.Reader(self.data["average"]) as infile:
            average: sf.Frame = infile.read()
        with IC.Reader(self.data["fluctuation"]) as infile:
            fluctuation: sf.Frame = infile.read()
        parameters: sf.Frame = sf.Frame.from_concat(
            [fluctuation[["I", "J", "r_IJ"]], average[["r_IJ",]]],
            axis=1,
            columns=["I", "J", "Kb", "b0"],
        )
        parameters: sf.Frame = parameters.assign["Kb"](
            self.boltzmann / np.square(parameters["Kb"])
        )
        return parameters

    def simulate(
        self,
        *,
        input_dir: Union[Path, str] = Path.home(),
        charmm_input: Union[Path, str] = "init_fluct.inp",
        executable: str = shutil.which("charmm"),
        residue: Union[Path, str] = "fluctmatch.rtf",
        topology: Union[Path, str] = "fluctmatch.xplor.psf",
        coordinates: Union[Path, str] = "fluctmatch.cor",
        trajectory: Union[Path, str] = "cg.dcd",
        version: int = 39,
    ) -> NoReturn:
        """Run the simulation using CHARMM.

        input_dir : Path or str
            Location of input files
        charmm_input : Path or str
            CHARMM input file
        executable : str
            Location of CHARMM executable file
        residue : Path or str
            CHARMM residue topology file (e.g., RTF)
        topology : Path or str
            CHARMM topology file (e.g., PSF or PDB)
        coordinates : Path or str
            Coordinate file
        trajectory : Path or str
            CHARMM trajectory file
        version : int
            CHARMM version
        """
        try:
            exec_file: Path = Path(executable)
        except TypeError:
            error_msg: str = (
                "Please set CHARMMEXEC with the location of your "
                "CHARMM executable file or add the charmm path to "
                "your PATH environment."
            )
            logger.exception(error_msg)
            raise OSError(error_msg)

        # Update data dictionary
        self.data.update(
            dict(
                residue=residue,
                topology=Path(input_dir) / topology,
                coordinates=Path(input_dir) / coordinates,
                trajectory=Path(input_dir) / trajectory,
                stream=Path(input_dir) / self.prefix.with_suffix(".stream"),
                version=version,
            )
        )

        # Read Jinja2 templates
        env: Environment = Environment(
            loader=PackageLoader("fluctmatch"), autoescape=True
        )
        header: Template = env.get_template("charmm_cg_header.j2")
        body: Template = env.get_template("charmm_fluctmatch.j2")
        with open(Path(input_dir) / charmm_input, "w") as infile:
            print(header.render(self.data) + body.render(self.data), file=infile)

        with open(self.logfile, "w") as logfile:
            subprocess.check_call(
                [exec_file, "-i", input_dir / charmm_input],
                stdout=logfile,
                stderr=subprocess.STDOUT,
            )

    def run(
        self,
        nma_exec: str = None,
        tol: float = 1.0e-4,
        min_cycles: int = 200,
        max_cycles: int = 200,
        force_tol: float = 0.02,
    ) -> NoReturn:
        pass

    def initialize(self, nma_exec: str = None, restart: bool = False) -> NoReturn:
        pass
