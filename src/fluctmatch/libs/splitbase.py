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
"""Base class for splitting trajectories.
"""

import abc
from pathlib import Path
from typing import NoReturn, Union

from class_registry import AutoRegister, ClassRegistry

splitter: ClassRegistry = ClassRegistry("split_type")


class SplitBase(abc.ABC, metaclass=AutoRegister(splitter)):
    """Create a smaller trajectory from a trajectory.

    Parameters
    ----------
    data_dir : str, optional
        Location of the main data directory
    exec_file : str, optional
        location of executable file
    """

    def __init__(
        self,
        data_dir: Union[Path, str, None] = Path.cwd() / "data",
        exec_file: Union[Path, str] = None,
    ):
        self._data_dir: Path = Path(data_dir)
        try:
            self._executable: Path = Path(exec_file)
        except TypeError:
            self._executable: Path = Path()

    @property
    def data(self) -> Path:
        """Location of the data directory"""
        return self._data_dir

    @data.setter
    def data(self, directory: Union[Path, str]) -> NoReturn:
        self._data_dir: Path = Path(directory)

    @property
    def executable(self) -> Path:
        """Location of executable file."""
        return self._executable

    @executable.setter
    def executable(self, filename: Union[Path, str]) -> NoReturn:
        self._executable: Path = Path(filename)

    @abc.abstractmethod
    def split(
        self, topology: Union[Path, str], trajectory: Union[Path, str], **kwargs: dict
    ) -> NoReturn:
        """Split a trajectory into smaller trajectories.

        Parameters
        ----------
        topology : str, optional
            Topology file or directory
        trajectory : str, optional
            A Gromacs trajectory file (e.g., xtc trr)
        start : int, optional
            Starting frame number of the trajectory
        stop : int, optional
            Ending frame number of the trajectory
        """
        return
