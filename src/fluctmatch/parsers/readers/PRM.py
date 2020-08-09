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
from collections import namedtuple
from pathlib import Path
from typing import ClassVar, Dict, List, Optional, Tuple, Union

import static_frame as sf

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

    format = "PRM"
    units: ClassVar[Dict[str, Optional[str]]] = dict(time=None, length="Angstrom")

    _headers = namedtuple("_neaders", ("ATOMS BONDS ANGLES DIHEDRALS IMPROPERS"))
    _columns = _headers(
        ATOMS=tuple("header type atom mass".split()),
        BONDS=tuple("I J Kb b0".split()),
        ANGLES=tuple("I J K Ktheta theta0 Kub S0".split()),
        DIHEDRALS=tuple("I J K L Kchi n delta".split()),
        IMPROPERS=tuple("I J K L Kchi n delta".split()),
    )
    _dtypes = _headers(
        ATOMS=dict(header=str, type=int, atom=str, mass=float),
        BONDS=dict(I=str, J=str, Kb=float, b0=float),
        ANGLES=dict(
            I=str, J=str, K=str, Ktheta=float, theta0=float, Kub=object, S0=object,
        ),
        DIHEDRALS=dict(I=str, J=str, K=str, L=str, Kchi=float, n=int, delta=float,),
        IMPROPERS=dict(I=str, J=str, K=str, L=str, Kchi=float, n=int, delta=float,),
    )

    def __init__(self, filename: Union[str, Path]) -> None:
        self.filename = Path(filename).with_suffix("." + self.format.lower())

    def read(self) -> namedtuple:
        """Parse the parameter file.

        Returns
        -------
        Dictionary with CHARMM parameters per key.
        """
        headers: Tuple[str, ...] = self._headers._fields
        buffers: Dict[str, List] = {_: [] for _ in self._headers._fields}

        with open(self.filename) as prmfile:
            for line in prmfile:
                line: str = line.strip()
                if line.startswith("*") or line.startswith("!") or not line:
                    continue  # ignore TITLE and empty lines

                # Parse sections
                if line in headers:
                    section = line
                    continue

                if (
                    line.startswith("NONBONDED")
                    or line.startswith("CMAP")
                    or line.startswith("END")
                    or line.startswith("end")
                ):
                    break

                buffers[section].append(line.split())

        parameters = self._headers(
            **{
                _: sf.Frame.from_records(
                    buffers[_],
                    columns=getattr(self._columns, _),
                    dtypes=getattr(self._dtypes, _),
                    name=_.lower(),
                )
                for _ in self._headers._fields
            }
        )
        if parameters.ATOMS.size > 0:
            parameters = parameters._replace(ATOMS=parameters.ATOMS.drop["header"])

        return parameters
