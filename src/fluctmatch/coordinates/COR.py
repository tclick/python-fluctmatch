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
"""CHARMM coordinate file reader and writer for viewing in VMD."""

import itertools
import logging
import warnings
from typing import Mapping, Optional, Any, List, Generator

import numpy as np
import MDAnalysis as mda
from MDAnalysis.coordinates import CRD
from MDAnalysis.exceptions import NoDataError
from MDAnalysis.lib import util

logger = logging.getLogger(__name__)


class CORReader(CRD.CRDReader):
    """CRD reader that implements the standard and extended CRD coordinate formats

    .. versionchanged:: 0.11.0
       Now returns a ValueError instead of FormatError.
       Frames now 0-based instead of 1-based.
    """
    format: str = "COR"
    units: Mapping[str, Optional[str]] = {"time": None, "length": "Angstrom"}

    def __init__(self, filename, convert_units=None, n_atoms=None, **kwargs):
        super().__init__(filename, convert_units, n_atoms, **kwargs)

    @staticmethod
    def parse_n_atoms(filename, **kwargs):
        with open(filename, "r") as crdfile:
            for linenum, line in enumerate(crdfile):
                if line.strip().startswith('*') or line.strip() == "":
                    continue  # ignore TITLE and empty lines
                fields = line.split()
                if len(fields) <= 2:
                    # should be the natoms line
                    n_atoms: int = int(fields[0])
                    break
        return n_atoms


class CORWriter(CRD.CRDWriter):
    """COR writer that implements the CHARMM CRD EXT coordinate format.

    This class supercedes the original class provided by MDAnalysis by
    writing only the EXT format regardless of the number of atoms in the
    universe.

    Requires the following attributes:
    - resids
    - resnames
    - names
    - chainIDs
    - tempfactors

    .. versionchanged:: 0.11.0
       Frames now 0-based instead of 1-based
    """
    format: str = "COR"
    units: Mapping[str, Optional[str]] = {"time": None, "length": "Angstrom"}

    fmt: Mapping[str, str] = dict(
        # crdtype = "extended"
        # fortran_format = "(2I10,2X,A8,2X,A8,3F20.10,2X,A8,2X,A8,F20.10)"
        ATOM_EXT=("{serial:10d}{totRes:10d}  {resname:<8.8s}  {name:<8.8s}"
                  "{pos[0]:20.10f}{pos[1]:20.10f}{pos[2]:20.10f}  "
                  "{chainID:<8.8s}  {resSeq:<8d}{tempfactor:20.10f}"),
        NUMATOMS_EXT="{0:10d}  EXT",
        # crdtype = "standard"
        # fortran_format = "(2I5,1X,A4,1X,A4,3F10.5,1X,A4,1X,A4,F10.5)"
        ATOM=("{serial:5d}{totRes:5d} {resname:<4.4s} {name:<4.4s}"
              "{pos[0]:10.5f}{pos[1]:10.5f}{pos[2]:10.5f} "
              "{chainID:<4.4s} {resSeq:<4d}{tempfactor:10.5f}"),
        TITLE="* FRAME {frame} FROM {where}",
        NUMATOMS="{0:5d}",
    )

    def __init__(self, filename: str, **kwargs):
        """
        Parameters
        ----------
        filename : str or :class:`~MDAnalysis.lib.util.NamedStream`
             name of the output file or a stream
        """
        super().__init__(filename, **kwargs)

        self.filename: str = util.filename(filename, ext="cor")
        self.crd: Optional[str] = None

    def write(self, selection: mda.Universe, frame: Optional[int]=None):
        """Write selection at current trajectory frame to file.

        write(selection,frame=FRAME)

        selection         MDAnalysis AtomGroup
        frame             optionally move to frame FRAME
        """
        u: mda.Universe = selection.universe
        if frame is not None:
            u.trajectory[frame]  # advance to frame
        else:
            try:
                frame = u.trajectory.ts.frame
            except AttributeError:
                frame = 0

        atoms: mda.AtomGroup = selection.atoms
        coor: np.ndarray = atoms.positions

        n_atoms: int = atoms.n_atoms
        # Detect which format string we"re using to output (EXT or not)
        # *len refers to how to truncate various things,
        # depending on output format!
        at_fmt: str = self.fmt["ATOM_EXT"]
        serial_len: int = 10
        resid_len: int = 8
        totres_len: int = 10

        # Check for attributes, use defaults for missing ones
        attrs: Mapping[str, Any] = {}
        missing_topology: List[Any] = []
        for attr, default in (
            ("resnames", itertools.cycle(("UNK",))),
                # Resids *must* be an array because we index it later
            ("resids", np.ones(n_atoms, dtype=np.int)),
            ("names", itertools.cycle(("X", ))),
            ("tempfactors", itertools.cycle((0.0,))),
        ):
            try:
                attrs[attr]: Any = getattr(atoms, attr)
            except (NoDataError, AttributeError):
                attrs[attr]: Any = default
                missing_topology.append(attr)
        # ChainIDs - Try ChainIDs first, fall back to Segids
        try:
            attrs["chainIDs"]: str = atoms.segids
        except (NoDataError, AttributeError):
            # try looking for segids instead
            try:
                attrs["chainIDs"]: str = atoms.chainIDs
            except (NoDataError, AttributeError):
                attrs["chainIDs"]: Generator = itertools.cycle(("", ))
                missing_topology.append(attr)
        if missing_topology:
            miss: str = ", ".join(missing_topology)
            warnings.warn("Supplied AtomGroup was missing the following "
                          "attributes: {miss}. These will be written with "
                          "default values.".format(miss=miss))
            logger.warning("Supplied AtomGroup was missing the following "
                           "attributes: {miss}. These will be written with "
                           "default values.".format(miss=miss))

        with open(self.filename, "w") as crd:
            # Write Title
            logger.info(f"Writing {self.filename}")
            print(self.fmt["TITLE"].format(frame=frame,
                                           where=u.trajectory.filename),
                  file=crd)
            print("*", file=crd)

            # Write NUMATOMS
            print(self.fmt["NUMATOMS_EXT"].format(n_atoms), file=crd)

            # Write all atoms
            current_resid: int = 1
            resids: List[int] = attrs["resids"]
            for i, pos, resname, name, chainID, resid, tempfactor in zip(
                    range(n_atoms), coor, attrs["resnames"], attrs["names"],
                    attrs["chainIDs"], attrs["resids"], attrs["tempfactors"]):
                if not i == 0 and resids[i] != resids[i - 1]:
                    current_resid += 1

                # Truncate numbers
                serial: int = int(str(i + 1)[-serial_len:])
                resid: int = int(str(resid)[-resid_len:])
                current_resid: int = int(str(current_resid)[-totres_len:])

                print(at_fmt.format(serial=serial, totRes=current_resid,
                                    resname=resname, name=name, pos=pos,
                                    chainID=chainID, resSeq=resid,
                                    tempfactor=tempfactor), file=crd)
            logger.info("Coordinate file successfully written.")
