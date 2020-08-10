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
"""Class to write CHARMM parameter files with a prm extension."""

import logging
import textwrap
from pathlib import Path
from typing import (Dict, NamedTuple, Optional, Tuple, Union)

import MDAnalysis as mda
import numpy as np
import static_frame as sf

from ..base import TopologyWriterBase

logger: logging.Logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class Writer(TopologyWriterBase):
    """Write a parameter dictionary to a CHARMM-formatted parameter file.

    Parameters
    ----------
    filename : str or :class:`~MDAnalysis.lib.util.NamedStream`
         name of the output file or a stream
    title : str
        Title lines at beginning of the file.
    charmm_version
        Version of CHARMM for formatting (default: 41)
    nonbonded
        Add the nonbonded section. (default: False)
    """

    format = "PRM"
    units: Dict[str, Optional[str]] = dict(time=None, length="Angstrom")

    _HEADERS: Tuple[str, ...] = (
        "ATOMS",
        "BONDS",
        "ANGLES",
        "DIHEDRALS",
        "IMPROPER",
    )
    _FORMAT: Dict[str, str] = dict(
        ATOMS="MASS %5d %-6s %9.5f",
        BONDS="%-6s %-6s %10.4f%10.4f",
        ANGLES="%-6s %-6s %-6s %8.2f%10.2f%10s%10s",
        DIHEDRALS="%-6s %-6s %-6s %-6s %12.4f%3d%9.2f",
        IMPROPER="%-6s %-6s %-6s %-6s %12.4f%3d%9.2f",
        NONBONDED="%-6s %5.1f %13.4f %10.4f",
    )

    def __init__(
        self,
        filename: Union[str, Path],
        *,
        charmm_version: int = 41,
        nonbonded: bool = False,
        n_atoms: Optional[int] = None,
    ) -> None:
        super().__init__()

        self.filename = Path(filename).with_suffix(".prm")
        self._version: int = charmm_version
        self._nonbonded: bool = nonbonded
        self.n_atoms: int = n_atoms

    def write(
        self, parameters: NamedTuple, atomgroup: Optional[mda.AtomGroup] = None,
    ):
        """Write a CHARMM-formatted parameter file.

        Parameters
        ----------
        parameters : dict
            Keys are the section names and the values are of class
            :class:`~pandas.DataFrame`, which contain the corresponding
            parameter data.
        atomgroup : :class:`~MDAnalysis.AtomGroup`, optional
            A collection of atoms in an AtomGroup to define the ATOMS section,
            if desired.
        """
        with open(self.filename, "w") as prmfile:
            print(textwrap.dedent(self.title).strip(), file=prmfile)
            print(file=prmfile)

            if self._version > 35 and parameters.ATOMS.size == 0:
                if atomgroup is not None:
                    atom_types: np.ndarray = (
                        atomgroup.types
                        if np.issubdtype(atomgroup.types.dtype, np.int)
                        else np.arange(atomgroup.n_atoms) + 1
                    )
                    atoms: np.ndarray = np.vstack(
                        (atom_types, atomgroup.types, atomgroup.masses)
                    )
                    parameters["ATOMS"]: sf.Frame = parameters["ATOMS"].assign["type"](
                        atoms.T
                    )
                else:
                    raise RuntimeError(
                        "Either define ATOMS parameter or "
                        "provide a MDAnalsys.AtomGroup"
                    )

            if self._version >= 39 and parameters.ATOMS.size > 0:
                parameters = parameters._replace(
                    ATOMS=parameters.ATOMS.assign["type"](-1)
                )

            for key in parameters._fields:
                section: sf.Frame = getattr(parameters, key)
                if self._version < 36 and key == "ATOMS":
                    continue
                if section.size == 0:
                    continue

                print(key, file=prmfile)
                np.savetxt(prmfile, section.values, fmt=self._FORMAT[key])
                print(file=prmfile)

            nb_header: str = (
                """
                NONBONDED nbxmod  5 atom cdiel shift vatom vdistance vswitch -
                cutnb 14.0 ctofnb 12.0 ctonnb 10.0 eps 1.0 e14fac 1.0 wmin 1.5
                """
            )
            print(textwrap.dedent(nb_header[1:])[:-1], file=prmfile)

            if self._nonbonded:
                if parameters.ATOMS.size > 0:
                    atom_list = parameters.ATOMS["atom"]
                else:
                    i = parameters.BONDS["I"]
                    index = parameters.BONDS["J"].size
                    j = parameters.BONDS["J"].relabel(np.arange(index) + index)
                    atom_list = (
                        sf.Series.from_concat([i, j])
                        .drop_duplicated(exclude_first=True)
                        .sort_values()
                    )
                atom_list = atom_list.to_frame().relabel(np.arange(atom_list.size))
                nb_list = sf.Frame(np.zeros((atom_list.size, 3)), index=atom_list.index)
                nb_list = atom_list.join_inner(
                    nb_list,
                    left_depth_level=0,
                    right_depth_level=0,
                    composite_index=False,
                )
                np.savetxt(
                    prmfile, nb_list.values, fmt=self._FORMAT["NONBONDED"], delimiter=""
                )
            print("\nEND", file=prmfile)
