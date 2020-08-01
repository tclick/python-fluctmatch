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
"""Write CHARMM stream file."""

import textwrap
from pathlib import Path
from typing import ClassVar, Dict, Mapping, Optional, Union

import MDAnalysis as mda
import numpy as np
import static_frame as sf

from .. import base as topbase


class Writer(topbase.TopologyWriterBase):
    """Write a stream file to define internal coordinates within CHARMM.

    Parameters
    ----------
    filename : str
        The filename to which the internal coordinate definitions are written.
    n_atoms : int, optional
        The number of atoms in the output trajectory.
    title
        A header section written at the beginning of the stream file.
        If no title is given, a default title will be written.
    """

    format: ClassVar[str] = "STREAM"
    units: Dict[str, Optional[str]] = dict(time=None, length="Angstrom")

    def __init__(self, filename: Union[str, Path], **kwargs: Mapping):
        super().__init__()

        self.filename: Path = Path(filename).with_suffix(".stream")
        self._version: int = kwargs.get("charmm_version", 41)

        width: int = 4 if self._version < 36 else 8
        if self._version >= 36:
            self.fmt: str = """
                IC EDIT
                DIST %-{}s %{}d %-{}s %-{}s %{}d %-{}s%{}.1f
                END
                """.format(
                *[width] * 7
            )
        else:
            self.fmt = """
                IC EDIT
                DIST BYNUM %{}d BYNUM %{}d %{}.1f
                END
                """.format(
                *[width] * 3
            )

    def write(self, universe: Union[mda.Universe, mda.AtomGroup]):
        """Write the bond information to a CHARMM-formatted stream file.

        Parameters
        ----------
        universe : :class:`~MDAnalysis.Universe` or :class:`~MDAnalysis.AtomGroup`
            A collection of atoms in a universe or atomgroup with bond
            definitions.
        """
        # Create the table
        try:
            n_bonds: int = universe.atoms.bonds.bonds()
            dist: np.ndarray = np.zeros_like(n_bonds, dtype=float)
            if self._version >= 36:
                atom1, atom2 = universe.atoms.bonds.atom1, universe.atoms.bonds.atom2
                data: sf.Frame = sf.Frame.from_dict(
                    dict(
                        segidI=atom1.segids,
                        resI=atom1.resids,
                        I=atom1.names,
                        segidJ=atom2.segids,
                        resJ=atom2.resids,
                        J=atom2.names,
                        b0=dist,
                    )
                )
            else:
                data: sf.Frame = sf.Frame(universe._topology.bonds.values)
                data: sf.Frame = data.from_concat(
                    [sf.Series(dist),]
                )
        except AttributeError:
            AttributeError("No bonds were found.")

        # Write the data to the file.
        with open(self.filename, "w") as stream_file:
            print(textwrap.dedent(self.title).strip(), file=stream_file)
            np.savetxt(
                stream_file, data.values, fmt=textwrap.dedent(self.fmt.strip("\n"))
            )
            print("RETURN", file=stream_file)
