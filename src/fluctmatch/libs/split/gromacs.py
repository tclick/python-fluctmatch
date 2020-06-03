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
"""Split a Gromacs trajectory file into a smaller trajectory."""
import logging
import shutil
import subprocess
import tempfile
from contextlib import ExitStack
from pathlib import Path
from typing import List, NoReturn, Union

from ..splitbase import SplitBase

logger: logging.Logger = logging.getLogger(__name__)


class Split(SplitBase):
    """Create a smaller trajectory from a Gromacs trajectory.

    Parameters
    ----------
    data_dir : str, optional
        Location of the main data directory
    exec_file : str, optional
        location of executable file
    """

    split_type: str = "gmx"

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
                msg: str = "Cannot find Gromacs executable file."
                logger.error(msg)
                raise RuntimeError(msg)

    def split(
        self, topology: Union[Path, str], trajectory: Union[Path, str], **kwargs: dict
    ) -> NoReturn:
        """Create a subtrajectory from a Gromacs trajectory.

        Parameters
        ----------
        topology : str, optional
            A CHARMM protein structure file
        trajectory : str, optional
            A CHARMM trajectory file (e.g., dcd)
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
        index : str, optional
            A Gromacs index file (e.g., ndx)
        system : int
            Atom selection from Gromacs index file (0 = System, 1 = Protein)
        """
        subdir: Path = self._data_dir / kwargs.get("subdir", "1")

        # Attempt to create the necessary subdirectory
        subdir.mkdir(parents=True, exist_ok=True)

        # Various filenames
        data = dict(
            topology=Path(topology),
            trajectory=Path(trajectory),
            version=kwargs.get("charmm_version", 41),
            start=kwargs.get("start", 1),
            stop=kwargs.get("stop", 10000),
            output=kwargs.get("outfile", subdir / "aa.xtc",),
            index=kwargs.get("index", None),
            system=kwargs.get("system", 1),
        )
        index: int = kwargs.get("index", None)
        outfile: Path = kwargs.get(
            "outfile", subdir / "aa.xtc",
        )
        logfile: Path = subdir / kwargs.get("logfile", Path.cwd() / "split.log")

        command: List = [
            self._executable,
            "trjconv",
            "-s",
            topology,
            "-f",
            trajectory,
            "-o",
            outfile,
            "-b",
            f"{kwargs.get('start', 1):d}",
            "-e",
            f"{kwargs.get('stop', 10000):d}",
        ]

        if data["index"] is not None:
            command.append(["-n", index])
        _, fpath = tempfile.mkstemp(text=True)
        fpath: Path = Path(fpath)

        with ExitStack() as stack:
            temp = stack.enter_context(open(fpath, mode="w+"))
            log = stack.enter_context(open(logfile, mode="w"))
            logger.info(
                "Writing trajectory to %s and writing Gromacs output to %s",
                outfile,
                logfile,
            )
            print(kwargs.get("system", 0), file=temp)
            temp.seek(0)
            subprocess.check_call(
                command, stdin=temp, stdout=log, stderr=subprocess.STDOUT
            )
        fpath.unlink()
