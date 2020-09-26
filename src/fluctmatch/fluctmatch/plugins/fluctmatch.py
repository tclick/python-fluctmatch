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
from typing import MutableMapping, Optional

import numpy as np
import static_frame as sf
from jinja2 import Environment, PackageLoader, Template

from .. import base
from ...parsers.readers import IC, PRM

logger: logging.Logger = logging.getLogger(__name__)


class FluctMatch(base.FluctMatchBase):
    def __init__(
        self,
        *,
        temperature: float = 300.0,
        output_dir: Optional[str] = None,
        prefix: Optional[str] = "fluctmatch",
    ) -> None:
        """Fluctuation matching using CHARMM.

        Parameters
        ----------
        output_dir, str, optional
            Output directory
        temperature : float
            Temperature (in K)
        prefix : str, optional
            Filename prefix
        """
        super(FluctMatch, self).__init__(
            output_dir=output_dir, temperature=temperature, prefix=prefix,
        )
        title = """
            * Calculate the bond distance average and fluctuations.
            *
        """
        self.data.update(dict(title=textwrap.dedent(title[1:]),))

    def calculate(self, *, target: Optional[sf.Frame] = None) -> sf.Frame:
        """Calculate the parameters from CHARMM fluctuations.

        target : Frame or None
            Target fluctuation data
        """
        error_columns = (
            "fluct_rms",
            "b0_rms",
            "Kb_rms",
        )
        with IC.Reader(self.data["average"]) as infile:
            average = infile.read()
        with IC.Reader(self.data["fluctuation"]) as infile:
            fluctuation = infile.read()
        parameters = sf.Frame.from_concat(
            [fluctuation[["I J r_IJ".split()]], average[["r_IJ",]]],
            axis=1,
            columns=["I J Kb b0".split()],
        )

        if target is None:
            parameters = parameters.assign["Kb"](
                self.BOLTZMANN / np.square(parameters["Kb"].values)
            )
        else:
            # Retrieve previously calculated parameters
            factor = self.BOLTZMANN * self.K_FACTOR
            with PRM.Reader(self.data["parameter"]) as infile:
                old_parameters: sf.Frame = infile.read()["BONDS"]

            target = target.apply["Kb"](np.power(target["Kb"].values, -2))
            optimized = parameters.apply["Kb"](np.power(parameters["Kb"].values, -2))
            optimized = optimized.apply["Kb"](factor * (optimized["Kb"] - target["Kb"]))
            optimized = old_parameters.apply["Kb"](
                old_parameters["Kb"] - optimized["Kb"]
            )
            optimized = optimized.apply["Kb"](
                np.clip(optimized["Kb"].values, a_min=0.0, a_max=None)
            )

        return parameters

    def simulate(
        self,
        *,
        input_dir: Optional[str] = None,
        charmm_input: Optional[str] = None,
        executable: Optional[str] = None,
        parameters: Optional[MutableMapping] = None,
        trajectory: Optional[str] = None,
        logfile: Optional[str] = None,
        version: int = 39,
        xplor: bool = True,
        initialize: bool = False,
        vibration: bool = False,
    ) -> None:
        """Run the simulation using CHARMM.

        input_dir : str, optional
            Location of input files
        charmm_input : str, optional
            CHARMM input file
        executable : str
            Location of CHARMM executable file
        logfile : str, optional
            File for CHARMM output
        trajectory : str, optional
            CHARMM trajectory file
        version : int
            CHARMM version
        xplor : bool
            XPLOR or standard CHARMM protein structure file
        initialize: bool
            Determine target values or run normal mode analysis
        vibration : bool
            Save normal mode data
        """
        try:
            exec_file = Path(
                shutil.which("charmm") if executable is None else executable
            )
        except TypeError:
            error_msg = (
                "Please set CHARMMEXEC with the location of your "
                "CHARMM executable file or add the charmm path to "
                "your PATH environment."
            )
            logger.exception(error_msg)
            raise OSError(error_msg)

        # Update data dictionary
        self.data["average"] = (
            self.output_dir / "init.average.ic"
            if initialize
            else self.output_dir / "average.ic"
        )
        self.data["fluctuation"] = (
            self.output_dir / "init.fluct.ic"
            if initialize
            else self.output_dir / "fluct.ic"
        )
        if not initialize:
            self.data["parameter"] = self.output_dir / self.prefix.with_suffix(".prm")
        if vibration:
            self.data["normal_modes"] = self.output_dir / self.prefix.with_suffix(
                ".vib"
            )
        topology = (
            self.prefix.with_suffix(".xplor.psf")
            if xplor
            else self.prefix.with_suffix(".psf")
        )
        indir = Path.home() if input_dir is None else Path(input_dir)
        self.data.update(
            dict(
                initialize=initialize,
                residue=indir / self.prefix.with_suffix(".rtf"),
                topology=indir / topology,
                coordinates=indir / self.prefix.with_suffix(".cor"),
                trajectory=indir / trajectory,
                stream=indir / self.prefix.with_suffix(".stream"),
                nma_coord=self.output_dir / self.prefix.with_suffix(".mini.cor"),
                version=version,
                vibrations=vibration,
            )
        )

        # Read Jinja2 templates
        env = Environment(loader=PackageLoader("fluctmatch"), autoescape=True)
        header: Template = env.get_template("charmm_cg_header.j2")
        body: Template = env.get_template("charmm_fluctmatch.j2")

        filename = indir / charmm_input
        with open(filename, "w") as infile:
            file_contents = header.render(self.data) + body.render(self.data)
            print(file_contents, file=infile)

        with open(self.output_dir / logfile, "w") as output:
            subprocess.check_call(
                [exec_file, "-i", filename], stdout=output, stderr=subprocess.STDOUT,
            )
