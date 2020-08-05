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
"""Split a CHARMM trajectory file into a smaller trajectory."""
import logging
import shutil
import subprocess
import textwrap
from pathlib import Path
from typing import Any, Dict, List, NoReturn, Union

from jinja2 import Environment, PackageLoader, Template

from ..splitbase import SplitBase

logger: logging.Logger = logging.getLogger(__name__)


class Split(SplitBase):
    """Create a smaller trajectory from a CHARMM trajectory.

    Parameters
    ----------
    data_dir : str, optional
        Location of the main data directory
    exec_file : str, optional
        location of executable file
    """

    split_type: str = "charmm"

    def __init__(
        self,
        data_dir: Union[Path, str] = Path.cwd() / "data",
        exec_file: Union[Path, str] = None,
    ):
        super().__init__(data_dir=data_dir, exec_file=exec_file)
        if not self._executable.is_file():
            try:
                self._executable: Path = Path(shutil.which(self.split_type))
            except TypeError:
                msg: str = "Cannot find CHARMM executable file."
                logger.error(msg)
                raise RuntimeError(msg)

    def split(
        self, topology: Union[Path, str], trajectory: Union[Path, str], **kwargs: dict
    ) -> NoReturn:
        """Create a subtrajectory from a CHARMM trajectory.

        Parameters
        ----------
        topology : str, optional
            A CHARMM protein structure file
        trajectory : str, optional
            A CHARMM trajectory file (e.g., dcd)
        toppar : str, optional
            Location of CHARMM topology/parameter files
        subdir : str, optional
            Subdirectory in data directory
        outfile : str, optional
            A CHARMM trajectory file (e.g., dcd)
        logfile : str, optional
            Log file for output of command
        start : int, optional
            Starting frame number
        stop : int, optional
            Final frame number
        charmm_version : int
            Version of CHARMM
        """
        title: str = """
            * Create a subtrajectory from a larger CHARMM trajectory.
            * This is for <= c35.
            *
        """
        suffixes: List[str] = [".PSF", ".COR", ".CRD", ".PDB"]
        subdir: Path = self._data_dir / kwargs.get("subdir", "1")

        # Attempt to create the necessary subdirectory
        subdir.mkdir(parents=True, exist_ok=True)

        # Various filenames
        data: Dict[str, Any] = dict(
            title=textwrap.dedent(title[1:]),
            topology=Path(topology),
            suffix=Path(topology).suffix.upper(),
            trajectory=Path(trajectory),
            version=kwargs.get("charmm_version", 41),
            toppar=Path(
                kwargs.get(
                    "toppar", Path(shutil.which("charmm")).parent / ".." / "toppar"
                )
            ),
            start=kwargs.get("start", 1),
            stop=kwargs.get("stop", 10000),
            output=kwargs.get("outfile", subdir / "aa.dcd",),
        )
        if data["suffix"] not in suffixes:
            raise ValueError(
                f"{topology} must have a PSF, CRD, COR, or PDB file extension."
            )

        if not data["topology"].exists():
            raise FileNotFoundError(f"{data['topology']} not found.")
        if not data["trajectory"].exists():
            raise FileNotFoundError(f"{data['trajectory']} not found.")
        input_file: Path = subdir / "split.inp"
        log_file: Path = kwargs.get("logfile", subdir / "split.log")

        with open(input_file, mode="w") as charmm_input:
            env: Environment = Environment(PackageLoader("fluctmatch"), autoescape=True)
            header: Template = env.get_template("charmm_aa_header.j2")
            body: Template = env.get_template("charmm_split.j2")
            print(header.render(**data) + body.render_async(**data), file=charmm_input)
        command = [self._executable, "-i", input_file, "-o", log_file]
        subprocess.check_call(command)
