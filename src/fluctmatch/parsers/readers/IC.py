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
"""Read CHARMM internal coordinate files."""

import logging
from pathlib import Path
from typing import ClassVar, Dict, Iterator, List, Optional, Union

import numpy as np
import static_frame as sf

from ..base import TopologyReaderBase

logger: logging.Logger = logging.getLogger(__name__)


class Reader(TopologyReaderBase):
    """
    Parameters
    ----------
    filename : str or :class:`~Path`
         name of the output file or a stream
    """

    format: ClassVar[str] = "IC"
    units: Dict[str, Optional[str]] = dict(time=None, length="Angstrom")

    _cols: Dict[str, List[str]] = dict(
        STANDARD=[
            "#",
            "resI",
            "I",
            "resJ",
            "J",
            "resK",
            "K",
            "resL",
            "L",
            "r_IJ",
            "T_IJK",
            "P_IJKL",
            "T_JKL",
            "r_KL",
        ],
        RESID=[
            "#",
            "segidI",
            "resI",
            "I",
            "segidJ",
            "resJ",
            "J",
            "segidK",
            "resK",
            "K",
            "segidL",
            "resL",
            "L",
            "r_IJ",
            "T_IJK",
            "P_IJKL",
            "T_JKL",
            "r_KL",
        ],
    )
    _dtypes: Dict[str, List[type]] = dict(
        STANDARD=[
            int,
            int,
            str,
            int,
            str,
            str,
            str,
            str,
            str,
            float,
            float,
            float,
            float,
            float,
        ],
        RESID=[
            int,
            str,
            int,
            str,
            str,
            int,
            str,
            str,
            str,
            str,
            str,
            str,
            str,
            float,
            float,
            float,
            float,
            float,
        ],
    )

    def __init__(self, filename: Union[str, Path]):
        self.filename: Path = Path(filename).with_suffix(".ic")

    def read(self) -> sf.Frame:
        """Read the internal coordinates file.

        Returns
        -------
        :class:`~pandas.DataFrame`
            An internal coordinates table.
        """
        with open(self.filename) as infile:
            logger.info("Reading %s", self.filename)

            # Read title and header lines
            for line in infile:
                line: str = line.split("!")[0].strip()
                if line.startswith("*") or line.startswith("!") or not line:
                    continue  # ignore TITLE, comments, and empty lines
                break
            line: List = list(map(int, line.split()))
            key: str = "RESID" if line[1] == 2 else "STANDARD"
            resid_a = line[1]

            n_lines, resid_b = list(map(int, next(infile).split()))
            if resid_a != resid_b:
                raise IOError("A mismatch has occurred on determining the IC format.")

            # Read the internal coordinates
            rows: List = []
            for line in infile:
                columns: Iterator = filter(lambda x: x != ":", line.split())
                rows.append(list(columns))
            table: sf.Frame = sf.Frame.from_records(
                rows, columns=self._cols[key], dtypes=self._dtypes, name="IC"
            )
            table: sf.Frame = table.set_index(table.columns[0], drop=True)

            if n_lines != table.shape[0]:
                raise IOError(
                    f"A mismatch has occurred between the number of "
                    f"lines expected and the number of lines read. "
                    f"({n_lines:d} != {len(table):d})"
                )

            logger.info("Table read successfully.")
        return table
