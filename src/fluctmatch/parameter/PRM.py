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
"""Classes to read and write CHARMM parameter files."""

import logging
import textwrap
import time
from io import StringIO
from os import environ
from pathlib import Path
from typing import (ClassVar, Dict, List, Mapping, MutableMapping, Optional,
                    Tuple, Union, TextIO)

import MDAnalysis as mda
import numpy as np
import pandas as pd
from MDAnalysis.lib.util import asiterable, iterable
from ..topology.base import TopologyReaderBase, TopologyWriterBase

logger: logging.Logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class ParamReader(TopologyReaderBase):
    """Read a CHARMM-formated parameter file.

    Parameters
    ----------
    filename : str or :class:`~MDAnalysis.lib.util.NamedStream`
         name of the output file or a stream
    """
    format: ClassVar[str] = "PRM"
    units: ClassVar[Dict[str, Optional[str]]] = dict(time=None,
                                                     length="Angstrom")

    _prmindex: ClassVar[Dict[str, np.ndarray]] = dict(
        ATOMS=np.arange(1, 4),
        BONDS=np.arange(4),
        ANGLES=np.arange(7),
        DIHEDRALS=np.arange(6))
    _prmcolumns: Dict[str, List[str]] = dict(
        ATOMS=["hdr", "type", "atom", "mass"],
        BONDS=["I", "J", "Kb", "b0"],
        ANGLES=["I", "J", "K", "Ktheta", "theta0", "Kub", "S0"],
        DIHEDRALS=["I", "J", "K", "L", "Kchi", "n", "delta"],
        IMPROPER=["I", "J", "K", "L", "Kchi", "n", "delta"])
    _dtypes: Dict[str, Dict] = dict(
        ATOMS=dict(hdr=np.str, type=np.int, atom=np.str, mass=np.float,),
        BONDS=dict(I=np.str, J=np.str, Kb=np.float, b0=np.float),
        ANGLES=dict(I=np.str, J=np.str, K=np.str, Ktheta=np.float, 
                    theta0=np.float, Kub=np.object, S0=np.object),
        DIHEDRALS=dict(I=np.str, J=np.str, K=np.str, L=np.str, Kchi=np.float, 
                       n=np.int, delta=np.float),
        IMPROPER=dict(I=np.str, J=np.str, K=np.str, L=np.str, Kchi=np.float, 
                      n=np.int, delta=np.float),
    )
    _na_values: ClassVar[Dict[str, Dict]] = dict(
        ATOMS=dict(type=-1, mass=0.0),
        BONDS=dict(Kb=0.0, b0=0.0),
        ANGLES=dict(Ktheta=0.0, theta0=0.0, Kub="", S0=""),
        DIHEDRALS=dict(Kchi=0.0, n=0, delta=0.0),
        IMPROPER=dict(Kchi=0.0, n=0, delta=0.0),
    )

    def __init__(self, filename: Union[str, Path]):
        self.filename = Path(filename).with_suffix(".prm")
        self._prmbuffers: Dict[str, TextIO] = dict(
            ATOMS=StringIO(),
            BONDS=StringIO(),
            ANGLES=StringIO(),
            DIHEDRALS=StringIO(),
            IMPROPER=StringIO(),
        )

    def read(self):
        """Parse the parameter file.

        Returns
        -------
        Dictionary with CHARMM parameters per key.
        """
        parameters: Dict[str, pd.DataFrame] = dict.fromkeys(self._prmbuffers.keys())
        headers: Tuple[str, ...] = ("ATOMS", "BONDS", "ANGLES", "DIHEDRALS", 
                                    "IMPROPER")
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
                elif (line.startswith("NONBONDED") or
                      line.startswith("CMAP") or
                      line.startswith("END") or
                      line.startswith("end")):
                    break

                print(line, file=self._prmbuffers[section])

        for key, value in parameters.items():
            self._prmbuffers[key].seek(0)
            parameters[key]: pd.DataFrame = pd.read_csv(
                self._prmbuffers[key], header=None, names=self._prmcolumns[key],
                skipinitialspace=True, delim_whitespace=True, comment="!",
                dtype=self._dtypes[key])
            parameters[key].fillna(self._na_values[key], inplace=True)
        if not parameters["ATOMS"].empty:
            parameters["ATOMS"].drop("hdr", axis=1, inplace=True)
        return parameters


class PARReader(ParamReader):
    """Read CHARMM parameter files with extension .par"""
    format: ClassVar[str] = "PAR"

    def __init__(self, filename: Union[str, Path]):
        super().__init__(filename)
        
        self.filename: Path = Path(filename).with_suffix(".par")


class ParamWriter(TopologyWriterBase):
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
    format: ClassVar[str] = "PRM"
    units: Dict[str, Optional[str]] = dict(time=None, length="Angstrom")

    _headers: Tuple[str, ...] = ("ATOMS", "BONDS", "ANGLES", "DIHEDRALS", 
                                 "IMPROPER")
    _fmt: Dict[str, str] = dict(
        ATOMS="MASS %5d %-6s %9.5f",
        BONDS="%-6s %-6s %10.4f%10.4f",
        ANGLES="%-6s %-6s %-6s %8.2f%10.2f%10s%10s",
        DIHEDRALS="%-6s %-6s %-6s %-6s %12.4f%3d%9.2f",
        IMPROPER="%-6s %-6s %-6s %-6s %12.4f%3d%9.2f",
        NONBONDED="%-6s %5.1f %13.4f %10.4f",
    )

    def __init__(self, filename: Union[str, Path], **kwargs: Mapping):
        super().__init__()
        
        self.filename: Path = Path(filename).with_suffix(".prm")
        self._version: int = kwargs.get("charmm_version", 41)
        self._nonbonded: bool = kwargs.get("nonbonded", False)

        date: str = time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime())
        user: str = environ["USER"]
        self._title: Tuple[str, ...] = kwargs.get(
            "title", (f"* Created by fluctmatch on {date}",
                      f"* User: {user}",))
        if not iterable(self._title):
            self._title = asiterable(self._title)

    def write(self, parameters: MutableMapping[str, pd.DataFrame],
              atomgroup: Optional[mda.AtomGroup]=None):
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
            for title in self._title:
                print(title, file=prmfile)
            print(file=prmfile)

            if self._version > 35 and parameters["ATOMS"].empty:
                if atomgroup:
                    atom_types: np.ndarray = (
                        atomgroup.types
                        if np.issubdtype(atomgroup.types.dtype, np.int)
                        else np.arange(atomgroup.n_atoms) + 1
                    )
                    atoms: np.rec.recarray = np.hstack(
                        (atom_types[:, np.newaxis],
                         atomgroup.types[:, np.newaxis],
                         atomgroup.masses[:, np.newaxis]))
                    parameters["ATOMS"]: pd.DataFrame = pd.DataFrame(
                        atoms, columns=parameters["ATOMS"].columns)
                else:
                    raise RuntimeError("Either define ATOMS parameter or "
                                       "provide a MDAnalsys.AtomGroup")

            if self._version >= 39 and not parameters["ATOMS"].empty:
                parameters["ATOMS"]["type"]: int = -1

            for key in self._headers:
                value: pd.DataFrame = parameters[key]
                if self._version < 35 and key == "ATOMS":
                    continue
                if value.empty:
                    continue
                if not value.empty:
                    print(key, file=prmfile)
                    np.savetxt(prmfile, value, fmt=self._fmt[key])
                    print(file=prmfile)

            nb_header: str = ("""
                NONBONDED nbxmod  5 atom cdiel shift vatom vdistance vswitch -
                cutnb 14.0 ctofnb 12.0 ctonnb 10.0 eps 1.0 e14fac 1.0 wmin 1.5
                """)
            print(textwrap.dedent(nb_header[1:])[:-1], file=prmfile)

            if self._nonbonded:
                atom_list: np.ndarray = np.concatenate(
                    (parameters["BONDS"]["I"].values,
                     parameters["BONDS"]["J"].values))
                atom_list: np.ndarray = np.unique(atom_list)[:, np.newaxis]
                nb_list: np.ndarray = np.zeros((atom_list.size, 3))
                nb_list: np.ndarray = np.hstack((atom_list, nb_list))
                np.savetxt(prmfile, nb_list, fmt=self._fmt["NONBONDED"],
                           delimiter="")
            print("\nEND", file=prmfile)


class PARWriter(ParamWriter):
    """Write parameters to CHARMM parameter file with .par extension."""

    format: ClassVar[str] = "PAR"

    def __init__(self, filename: Union[str, Path], **kwargs: Mapping):
        super().__init__(filename, **kwargs)

        self.filename: Path = Path(filename).with_suffix(".par")
