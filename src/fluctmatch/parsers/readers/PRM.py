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
"""Class to read CHARMM parameter files."""

import logging
from io import StringIO
from pathlib import Path
from typing import ClassVar, Dict, List, Optional, TextIO, Tuple, Union

import numpy as np
import pandas as pd

from ..base import TopologyReaderBase

logger: logging.Logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class Reader(TopologyReaderBase):
    """Read a CHARMM-formated parameter file.

    Parameters
    ----------
    filename : str or :class:`~MDAnalysis.lib.util.NamedStream`
         name of the output file or a stream
    """

    format: ClassVar[str] = "PRM"
    units: ClassVar[Dict[str, Optional[str]]] = dict(time=None, length="Angstrom")

    _prmindex: ClassVar[Dict[str, np.ndarray]] = dict(
        ATOMS=np.arange(1, 4),
        BONDS=np.arange(4),
        ANGLES=np.arange(7),
        DIHEDRALS=np.arange(6),
    )
    _prmcolumns: ClassVar[Dict[str, List[str]]] = dict(
        ATOMS=["hdr", "type", "atom", "mass"],
        BONDS=["I", "J", "Kb", "b0"],
        ANGLES=["I", "J", "K", "Ktheta", "theta0", "Kub", "S0"],
        DIHEDRALS=["I", "J", "K", "L", "Kchi", "n", "delta"],
        IMPROPER=["I", "J", "K", "L", "Kchi", "n", "delta"],
    )
    _dtypes: ClassVar[Dict[str, Dict]] = dict(
        ATOMS=dict(hdr=np.str, type=np.int, atom=np.str, mass=np.float),
        BONDS=dict(I=np.str, J=np.str, Kb=np.float, b0=np.float),
        ANGLES=dict(
            I=np.str,
            J=np.str,
            K=np.str,
            Ktheta=np.float,
            theta0=np.float,
            Kub=np.object,
            S0=np.object,
        ),
        DIHEDRALS=dict(
            I=np.str,
            J=np.str,
            K=np.str,
            L=np.str,
            Kchi=np.float,
            n=np.int,
            delta=np.float,
        ),
        IMPROPER=dict(
            I=np.str,
            J=np.str,
            K=np.str,
            L=np.str,
            Kchi=np.float,
            n=np.int,
            delta=np.float,
        ),
    )
    _na_values: ClassVar[Dict[str, Dict]] = dict(
        ATOMS=dict(type=-1, mass=0.0),
        BONDS=dict(Kb=0.0, b0=0.0),
        ANGLES=dict(Ktheta=0.0, theta0=0.0, Kub="", S0=""),
        DIHEDRALS=dict(Kchi=0.0, n=0, delta=0.0),
        IMPROPER=dict(Kchi=0.0, n=0, delta=0.0),
    )

    def __init__(self, filename: Union[str, Path]):
        self.filename: Path = Path(filename).with_suffix(".prm")
        self._prmbuffers: Dict[str, TextIO] = dict(
            ATOMS=StringIO(),
            BONDS=StringIO(),
            ANGLES=StringIO(),
            DIHEDRALS=StringIO(),
            IMPROPER=StringIO(),
        )

    def read(self) -> Dict:
        """Parse the parameter file.

        Returns
        -------
        Dictionary with CHARMM parameters per key.
        """
        parameters: Dict[str, pd.DataFrame] = dict.fromkeys(self._prmbuffers.keys())
        headers: Tuple[str, ...] = (
            "ATOMS",
            "BONDS",
            "ANGLES",
            "DIHEDRALS",
            "IMPROPER",
        )
        section: str

        with open(self.filename) as prmfile:
            for line in prmfile:
                line: str = line.strip()
                if line.startswith("*") or line.startswith("!") or not line:
                    continue  # ignore TITLE and empty lines

                # Parse sections
                if line in headers:
                    section: str = line
                    continue

                if (
                    line.startswith("NONBONDED")
                    or line.startswith("CMAP")
                    or line.startswith("END")
                    or line.startswith("end")
                ):
                    break

                print(line, file=self._prmbuffers[section])

        for key, _ in parameters.items():
            self._prmbuffers[key].seek(0)
            parameters[key]: pd.DataFrame = pd.read_csv(
                self._prmbuffers[key],
                header=None,
                names=self._prmcolumns[key],
                skipinitialspace=True,
                delim_whitespace=True,
                comment="!",
                dtype=self._dtypes[key],
            )
            parameters[key]: pd.DataFrame = parameters[key].fillna(self._na_values[key])
        if not parameters["ATOMS"].empty:
            parameters["ATOMS"]: pd.DataFrame = parameters["ATOMS"].drop("hdr", axis=1)
        return parameters
