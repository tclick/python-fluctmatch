# -*- coding: utf-8 -*-
#
#  python-fluctmatch -
#  Copyright (c) 2019 Timothy H. Click, Ph.D.
#
#  All rights reserved.
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:
#
#  Redistributions of source code must retain the above copyright notice, this
#  list of conditions and the following disclaimer.
#
#  Redistributions in binary form must reproduce the above copyright notice,
#  this list of conditions and the following disclaimer in the documentation
#  and/or other materials provided with the distribution.
#
#  Neither the name of the author nor the names of its contributors may be used
#  to endorse or promote products derived from this software without specific
#  prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS “AS IS”
#  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
#  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
#  ARE DISCLAIMED. IN NO EVENT SHALL THE REGENTS OR CONTRIBUTORS BE LIABLE FOR
#  ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
#  DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
#  SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
#  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
#  OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
#  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
#  Timothy H. Click, Nixon Raj, and Jhih-Wei Chu.
#  Simulation. Meth Enzymology. 578 (2016), 327-342,
#  Calculation of Enzyme Fluctuograms from All-Atom Molecular Dynamics
#  doi:10.1016/bs.mie.2016.05.024.

import logging
import textwrap
from pathlib import Path
from typing import ClassVar, Dict, Mapping, Optional, Union

import numpy as np
import pandas as pd
from MDAnalysis.lib.util import FORTRANReader

from ..topology.base import (TopologyReaderBase, TopologyWriterBase)

logger: logging.Logger = logging.getLogger(__name__)


class IntcorReader(TopologyReaderBase):
    """
    Parameters
    ----------
    filename : str or :class:`~Path`
         name of the output file or a stream
    """
    format: ClassVar[str] = "IC"
    units: Dict[str, Optional[str]] = dict(time=None, length="Angstrom")

    fmt: Dict[str, str] = dict(
        # fortran_format = "(I5,1X,4(I3,1X,A4),F9.4,3F8.2,F9.4)"
        STANDARD="I5,1X,I3,1X,A4,I3,1X,A4,I3,1X,A4,I3,1X,A4,F9.4,3F8.2,F9.4",
        # fortran_format = "(I9,1X,4(I5,1X,A8),F9.4,3F8.2,F9.4)"
        EXTENDED="I9,1X,I5,1X,A8,I5,1X,A8,I5,1X,A8,I5,1X,A8,F9.4,3F8.2,F9.4",
        # fortran_format = "(I5,4(1X,A4,1X,A4,1X,A4,"":""),F12.6,3F12.4,F12.6)"
        STANDARD_RESID=(
            "I5,1X,A4,1X,A4,1X,A4,A1,1X,A4,1X,A4,1X,A4,A1,1X,A4,1X,A4,1X,A4,A1,"
            "1X,A4,1X,A4,1X,A4,A1,F12.6,3F12.4,F12.6"
        ),
        # fortran_format = "(I10,4(1X,A8,1X,A8,1X,A8,"":""),F12.6,3F12.4,F12.6)"
        EXTENDED_RESID=(
            "I10,1X,A8,1X,A8,1X,A8,A1,1X,A8,1X,A8,1X,A8,A1,1X,A8,1X,A8,1X,A8,"
            "A1,1X,A8,1X,A8,1X,A8,A1,F12.6,3F12.4,F12.6"
        ),
    )
    cols: np.ndarray = np.asarray([
        "segidI", "resI", "I", "segidJ", "resJ", "J", "segidK", "resK", "K",
        "segidL", "resL", "L", "r_IJ", "T_IJK", "P_IJKL", "T_JKL", "r_KL"
    ])

    def __init__(self, filename: Union[str, Path]):
        self.filename: Path = Path(filename).with_suffix(".ic")

    def read(self) -> pd.DataFrame:
        """Read the internal coordinates file.

        Returns
        -------
        :class:`~pandas.DataFrame`
            An internal coordinates table.
        """
        table: pd.DataFrame = pd.DataFrame()
        with open(self.filename) as infile:
            logger.info(f"Reading {self.filename}")

            # Read title and header lines
            for line in infile:
                line: str = line.split("!")[0].strip()
                if line.startswith("*") or not line:
                    continue  # ignore TITLE and empty lines
                break
            line: np.ndarray = np.fromiter(line.strip().split(), dtype=np.int)
            key: str = "EXTENDED" if line[0] == 30 else "STANDARD"
            key += "_RESID" if line[1] == 2 else ""
            resid_a = line[1]

            line: str = next(infile).strip().split()
            n_lines, resid_b = np.array(line, dtype=np.int)
            if resid_a != resid_b:
                raise IOError(
                    "A mismatch has occurred on determining the IC format."
                )

            # Read the internal coordinates
            TableParser = FORTRANReader(self.fmt[key])
            table: pd.DataFrame = pd.DataFrame(
                [TableParser.read(_) for _ in infile], dtype=np.object
            ).set_index(0)
            table: pd.DataFrame = table[table != ":"].dropna(axis=1)
            table: pd.DataFrame = table.apply(pd.to_numeric, errors="ignore")
            if n_lines != table.shape[0]:
                raise IOError(
                    f"A mismatch has occurred between the number of "
                    f"lines expected and the number of lines read. "
                    f"({n_lines:d} != {len(table):d})"
                )

            if key == "STANDARD":
                idx: tuple = np.where(
                    (self.cols != "segidI") & (self.cols != "segidJ") &
                    (self.cols != "segidK") & (self.cols != "segidL")
                )
                columns: np.ndarray = self.cols[idx]
            else:
                columns: np.ndarray = self.cols
            table.columns: pd.Index = columns
            logger.info("Table read successfully.")
        return table


class IntcorWriter(TopologyWriterBase):
    """Write a CHARMM-formatted internal coordinate file.

    Parameters
    ----------
    filename : str or :class:`Path`
        Filename for output.
    n_atoms : int, optional
        The number of atoms in the output trajectory.
    extended : bool, optional
        Format with wider columns than the standard column width.
    resid : bool, optional
        Include segment names within each atom definition.
    title : str or list of str, optional
        A header section written at the beginning of the stream file.
        If no title is given, a default title will be written.
    """
    format: ClassVar[str] = "IC"
    units: ClassVar[Dict[str, Optional[str]]] = dict(time="picosecond",
                                                     length="Angstrom")

    fmt: ClassVar[Dict[str, str]] = dict(
        # fortran_format = "(I5,1X,4(I3,1X,A4),F9.4,3F8.2,F9.4)"
        STANDARD=(
            "%5d %3s %-4s%3s %-4%3s %-4%3s %-4%9.4f%8.2f%8.2f%8.2f%9.4f"
        ),
        # fortran_format = "(I9,1X,4(I5,1X,A8),F9.4,3F8.2,F9.4)"
        EXTENDED=(
            "%10d %5s %-8s%5s %-8s%5s %-8s%5s %-8s%9.4f%8.2f%8.2f%8.2f%9.4f"
        ),
        # fortran_format = "(I5,4(1X,A4,1X,A4,1X,A4,"":""),F12.6,3F12.4,F12.6)"
        STANDARD_RESID=(
            "%5d %-4s %-4s %-4s: %-4s %-4s %-4s: %-4s %-4s %-4s: %-4s %-4s "
            "%-4s:%12.6f%12.4f%12.4f%12.4f%12.6f"
        ),
        # fortran_format = "(I10,4(1X,A8,1X,A8,1X,A8,"":""),F12.6,3F12.4,F12.6)"
        EXTENDED_RESID=(
            "%10d %-8s %-8s %-8s: %-8s %-8s %-8s: %-8s %-8s %-8s: %-8s %-8s "
            "%-8s:%12.6f%12.4f%12.4f%12.4f%12.6f"
        ),
    )

    def __init__(self, filename: Union[str, Path], **kwargs: Mapping):
        super().__init__()

        self.filename: Path = Path(filename).with_suffix(".ic")
        self._intcor: pd.DataFrame = None
        self._extended: bool = kwargs.get("extended", True)
        self._resid: bool = kwargs.get("resid", True)
        self.key: str = "EXTENDED" if self._extended else "STANDARD"
        self.key += "_RESID" if self._resid else ""

    def write(self, table: pd.DataFrame):
        """Write an internal coordinates table.

        Parameters
        ----------
        table : :class:`~pandas.DataFrame`
            A CHARMM-compliant internal coordinate table.
        """
        ictable: pd.DataFrame = table.copy()

        # Increment index.
        if ictable.index[0] == 0:
            ictable.index += 1

        with open(self.filename, "w") as outfile:
            logger.info(f"Writing to {self.filename}")
            # Save the title lines
            print(textwrap.dedent(self.title).strip(), file=outfile)

            # Save the header information
            line: np.ndarray = np.zeros((1, 20), dtype=np.int)
            line[0, 0]: int = 30 if self._extended else 20
            line[0, 1]: int = 2 if self._resid else 1
            np.savetxt(outfile, line, fmt="%4d", delimiter="")

            # Save the internal coordinates
            line: np.ndarray = np.zeros((1, 2), dtype=np.int)
            n_rows, _ = ictable.shape
            line[0, 0] += n_rows
            line[0, 1] += 2 if self._resid else 1
            np.savetxt(outfile, line, fmt="%5d", delimiter="")
            np.savetxt(outfile, ictable.reset_index().values,
                       fmt=self.fmt[self.key])
            logger.info("Table successfully written.")
