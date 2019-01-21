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
"""CHARMM PSF reader (for CHARMM36 style) and writer."""

import logging
import time
from os import environ
from typing import (Callable, ClassVar, List, Mapping,
                    Optional, TextIO, Tuple, Union)

import numpy as np
import pandas as pd
import MDAnalysis as mda
from MDAnalysis.lib import util
from MDAnalysis.lib.util import FORTRANReader
from MDAnalysis.topology import PSFParser
from MDAnalysis.topology.base import change_squash
from MDAnalysis.core.topologyattrs import (
    Atomids, Atomnames, Atomtypes, Masses, Charges, Resids, Resnums, Resnames,
    Segids, Bonds, Angles, Dihedrals, Impropers)
from MDAnalysis.core.topology import Topology

from . import base

logger: logging.Logger = logging.getLogger(__name__)


# Changed the segid squash_by to change_squash to prevent segment ID sorting.
class PSF36Parser(PSFParser.PSFParser):
    """Read topology information from a CHARMM/NAMD/XPLOR PSF_ file.

    Creates a Topology with the following Attributes:
    - ids
    - names
    - types
    - masses
    - charges
    - resids
    - resnames
    - segids
    - bonds
    - angles
    - dihedrals
    - impropers

    .. _PSF: http://www.charmm.org/documentation/c35b1/struct.html
    """
    format: ClassVar[str] = "PSF"

    def parse(self, **kwargs) -> Topology:
        """Parse PSF file into Topology

        Returns
        -------
        MDAnalysis *Topology* object
        """
        # Open and check psf validity
        with open(self.filename) as psffile:
            header: str = next(psffile)
            if not header.startswith("PSF"):
                err: str = (f"{self.filename} is not valid PSF file "
                            f"(header = {header})")
                logger.error(err)
                raise ValueError(err)
            header_flags: List[str] = header[3:].split()

            if "NAMD" in header_flags:
                self._format: str = "NAMD"  # NAMD/VMD
            elif "EXT" in header_flags:
                self._format: str = "EXTENDED"  # CHARMM
            else:
                self._format: str = "STANDARD"  # CHARMM
            if "XPLOR" in header_flags:
                self._format += "_XPLOR"

            next(psffile)
            title: str = next(psffile).split()
            if not (title[1] == "!NTITLE"):
                err: str = f"{psffile.name} is not a valid PSF file"
                logger.error(err)
                raise ValueError(err)
            # psfremarks = [psffile.next() for i in range(int(title[0]))]
            for _ in range(int(title[0])):
                next(psffile)
            logger.info(f"PSF file {psffile.name}: format {self._format}")

            # Atoms first and mandatory
            top: Topology = self._parse_sec(psffile, ('NATOM', 1,
                                                      1, self._parseatoms))
            # Then possibly other sections
            sections: Tuple[Tuple] = (
                # ("atoms", ("NATOM", 1, 1, self._parseatoms)),
                (Bonds, ("NBOND", 2, 4, self._parsesection)),
                (Angles, ("NTHETA", 3, 3, self._parsesection)),
                (Dihedrals, ("NPHI", 4, 2, self._parsesection)),
                (Impropers, ("NIMPHI", 4, 2, self._parsesection)),
                # ("donors", ("NDON", 2, 4, self._parsesection)),
                # ("acceptors", ("NACC", 2, 4, self._parsesection))
            )

            try:
                for attr, info in sections:
                    next(psffile)
                    top.add_TopologyAttr(attr(self._parse_sec(psffile, info)))
            except StopIteration:
                # Reached the end of the file before we expected
                pass

        return top

    def _parseatoms(self, lines: Callable[[TextIO], str], atoms_per: int, 
                    numlines: int) -> Topology:
        """Parses atom section in a Charmm PSF file.

        Normal (standard) and extended (EXT) PSF format are
        supported. CHEQ is supported in the sense that CHEQ data is simply
        ignored.


        CHARMM Format from ``source/psffres.src``:

        CHEQ::
          II,LSEGID,LRESID,LRES,TYPE(I),IAC(I),CG(I),AMASS(I),IMOVE(I),ECH(I),EHA(I)

          standard format:
            (I8,1X,A4,1X,A4,1X,A4,1X,A4,1X,I4,1X,2G14.6,I8,2G14.6)
            (I8,1X,A4,1X,A4,1X,A4,1X,A4,1X,A4,1X,2G14.6,I8,2G14.6)  XPLOR
          expanded format EXT:
            (I10,1X,A8,1X,A8,1X,A8,1X,A8,1X,I4,1X,2G14.6,I8,2G14.6)
            (I10,1X,A8,1X,A8,1X,A8,1X,A8,1X,A4,1X,2G14.6,I8,2G14.6) XPLOR

        no CHEQ::
          II,LSEGID,LRESID,LRES,TYPE(I),IAC(I),CG(I),AMASS(I),IMOVE(I)

         standard format:
            (I8,1X,A4,1X,A4,1X,A4,1X,A4,1X,I4,1X,2G14.6,I8)
            (I8,1X,A4,1X,A4,1X,A4,1X,A4,1X,A4,1X,2G14.6,I8)  XPLOR
          expanded format EXT:
            (I10,1X,A8,1X,A8,1X,A8,1X,A8,1X,I4,1X,2G14.6,I8)
            (I10,1X,A8,1X,A8,1X,A8,1X,A8,1X,A4,1X,2G14.6,I8) XPLOR

        NAMD PSF

        space separated, see release notes for VMD 1.9.1, psfplugin at
        http://www.ks.uiuc.edu/Research/vmd/current/devel.html :

        psfplugin: Added more logic to the PSF plugin to determine cases where
        the CHARMM "EXTended" PSF format cannot accomodate long atom types, and
        we add a "NAMD" keyword to the PSF file flags line at the top of the
        file. Upon reading, if we detect the "NAMD" flag there, we know that it
        is possible to parse the file correctly using a simple space-delimited
        scanf() format string, and we use that strategy rather than holding to
        the inflexible column-based fields that are a necessity for
        compatibility with CHARMM, CNS, X-PLOR, and other formats. NAMD and the
        psfgen plugin already assume this sort of space-delimited formatting,
        but that's because they aren't expected to parse the PSF variants
        associated with the other programs. For the VMD PSF plugin, having the
        "NAMD" tag in the flags line makes it absolutely clear that we're
        dealing with a NAMD-specific file so we can take the same approach.
        """
        # how to partition the line into the individual atom components
        atom_parsers: Mapping[str, str] = dict(
            STANDARD="I8,1X,A4,1X,A4,1X,A4,1X,A4,1X,I4,1X,2F14.6,I8",
            STANDARD_XPLOR="'(I8,1X,A4,1X,A4,1X,A4,1X,A4,1X,A4,1X,2F14.6,I8",
            EXTENDED="I10,1X,A8,1X,A8,1X,A8,1X,A8,1X,I4,1X,2F14.6,I8",
            EXTENDED_XPLOR="I10,1X,A8,1X,A8,1X,A8,1X,A8,1X,A6,1X,2F14.6,I8",
            NAMD="I8,1X,A4,1X,A4,1X,A4,1X,A4,1X,I4,1X,2F14.6,I8",
        )
        atom_parser: FORTRANReader = FORTRANReader(atom_parsers[self._format])

        # Allocate arrays
        atomids: np.ndarray = np.zeros(numlines, dtype=np.int32)
        segids: np.ndarray = np.zeros(numlines, dtype=object)
        resids: np.ndarray = np.zeros(numlines, dtype=np.int32)
        resnames: np.ndarray = np.zeros(numlines, dtype=object)
        atomnames: np.ndarray = np.zeros(numlines, dtype=object)
        atomtypes: np.ndarray = np.zeros(numlines, dtype=object)
        charges: np.ndarray = np.zeros(numlines, dtype=np.float32)
        masses: np.ndarray = np.zeros(numlines, dtype=np.float64)

        for i in range(numlines):
            try:
                line: str = lines()
            except StopIteration:
                err: str = f"{self.filename} is not valid PSF file"
                logger.error(err)
                raise ValueError(err)
            try:
                vals: List[str] = atom_parser.read(line)
            except ValueError:
                # last ditch attempt: this *might* be a NAMD/VMD
                # space-separated "PSF" file from VMD version < 1.9.1
                try:
                    atom_parser: FORTRANReader = FORTRANReader(
                        atom_parsers['NAMD'])
                    vals: List[str] = atom_parser.read(line)
                    logger.warning("Guessing that this is actually a NAMD-type "
                                   "PSF file... continuing with fingers "
                                   "crossed!")
                    logger.info("First NAMD-type line: {0}: {1}"
                                "".format(i, line.rstrip()))
                except ValueError:
                    atom_parser: FORTRANReader = FORTRANReader(
                        atom_parsers[self._format].replace("A6", "A4"))
                    vals: List[str] = atom_parser.read(line)
                    logger.warning("Guessing that this is actually a pre "
                                   "CHARMM36 PSF file... continuing with "
                                   "fingers crossed!")
                    logger.info(f"First NAMD-type line: {i}: {line.rstrip()}")

            atomids[i]: int = vals[0]
            segids[i]: str = vals[1] if vals[1] else "SYSTEM"
            resids[i]: int = vals[2]
            resnames[i]: str = vals[3]
            atomnames[i]: str = vals[4]
            atomtypes[i]: Union[int, str] = vals[5]
            charges[i]: float = vals[6]
            masses[i]: float = vals[7]

        # Atom
        atomids: Atomids = Atomids(atomids - 1)
        atomnames: Atomnames = Atomnames(atomnames)
        atomtypes: Atomtypes = Atomtypes(atomtypes)
        charges: Charges = Charges(charges)
        masses: Masses = Masses(masses)

        # Residue
        # resids, resnames
        residx, (new_resids, new_resnames, perres_segids) = change_squash(
            (resids, resnames, segids), (resids, resnames, segids))
        # transform from atom:Rid to atom:Rix
        residueids: Resids = Resids(new_resids)
        residuenums: Resnums = Resnums(new_resids.copy())
        residuenames: Resnames = Resnames(new_resnames)

        # Segment
        segidx, (perseg_segids, ) = change_squash((perres_segids, ),
                                                  (perres_segids, ))
        segids: Segids = Segids(perseg_segids)

        top: Topology = Topology(
            len(atomids), len(new_resids), len(segids),
            attrs=[atomids, atomnames, atomtypes, charges, masses, residueids,
                   residuenums, residuenames, segids],
            atom_resindex=residx,
            residue_segindex=segidx)

        return top


class PSFWriter(base.TopologyWriterBase):
    """PSF writer that implements the CHARMM PSF topology format.

    Requires the following attributes to be present:
    - ids
    - names
    - types
    - masses
    - charges
    - resids
    - resnames
    - segids
    - bonds

    .. versionchanged:: 3.0.0
       Uses numpy arrays for bond, angle, dihedral, and improper outputs.

    Parameters
    ----------
    filename : str or :class:`~MDAnalysis.lib.util.NamedStream`
         name of the output file or a stream
    n_atoms : int, optional
        The number of atoms in the output trajectory.
    extended
         extended format
    cmap
         include CMAP section
    cheq
         include charge equilibration
    title
         title lines at beginning of the file
    charmm_version
        Version of CHARMM for formatting (default: 41)
    """
    format: ClassVar[str] = "PSF"
    units: Mapping[str, Optional[str]] = dict(time=None, length=None)
    _fmt: Mapping[str, str] = dict(
        STD="%8d %-4s %-4d %-4s %-4s %4d %14.6f%14.6f%8d",
        STD_XPLOR="{%8d %4s %-4d %-4s %-4s %-4s %14.6f%14.6f%8d",
        STD_XPLOR_C35="%4d %-4s %-4d %-4s %-4s %-4s %14.6f%14.6f%8d",
        EXT="%10d %-8s %8d %-8s %-8s %4d %14.6f%14.6f%8d",
        EXT_XPLOR="%10d %-8s %-8d %-8s %-8s %-6s %14.6f%14.6f%8d",
        EXT_XPLOR_C35="%10d %-8s %-8d %-8s %-8s %-4s %14.6f%14.6f%8d")

    def __init__(self, filename, **kwargs):
        super().__init__()
        
        self.filename: str = util.filename(filename, ext="psf")
        self._extended: bool = kwargs.get("extended", True)
        self._cmap: bool = kwargs.get("cmap", True)
        self._cheq: bool = kwargs.get("cheq", True)
        self._version: int = kwargs.get("charmm_version", 41)
        self._universe: mda.Universe = None
        self._fmtkey: str = "EXT" if self._extended else "STD"

        date: str = time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime())
        user: str = environ["USER"]
        self._title: str = kwargs.get(
            "title", (f"* Created by fluctmatch on {date}",
                      f"* User: {user}",))
        if not util.iterable(self._title):
            self._title = util.asiterable(self._title)

        self.col_width: int = 10 if self._extended else 8
        self.sect_hdr: str = "{:>10d} !{}" if self._extended else "{:>8d} !{}"
        self.sect_hdr2: str = ("{:>10d}{:>10d} !{}"
                               if self._extended else "{:>8d}{:>8d} !{}")
        self.sections: Tuple[Tuple[str, str, int], ...] = (
            ("bonds", "NBOND: bonds", 8),
            ("angles", "NTHETA: angles", 9),
            ("dihedrals", "NPHI: dihedrals", 8),
            ("impropers", "NIMPHI: impropers", 8),
            ("donors", "NDON: donors", 8),
            ("acceptors", "NACC: acceptors", 8))

    def write(self, universe: Union[mda.Universe, mda.AtomGroup]):
        """Write universe to PSF format.

        Parameters
        ----------
        universe : :class:`~MDAnalysis.Universe` or :class:`~MDAnalysis.AtomGroup`
            A collection of atoms in a universe or atomgroup with bond
            definitions.
        """
        try:
            self._universe: Union[mda.Universe, mda.AtomGroup] = universe.copy()
        except TypeError:
            self._universe: mda.Universe(universe.filename,
                                         universe.trajectory.filename)
        xplor: bool = not np.issubdtype(universe.atoms.types.dtype, 
                                        np.signedinteger)

        header: str = "PSF"
        if self._extended:
            header += " EXT"
        if self._cheq:
            header += " CHEQ"
        if xplor:
            header += " XPLOR"
        if self._cmap:
            header += " CMAP"
        header += ""

        if xplor:
            self._fmtkey += "_XPLOR"
            if self._version < 36:
                self._fmtkey += "_C35"

        with open(self.filename, "w") as psffile:
            print(header, file=psffile)
            print(file=psffile)
            n_title: int = len(self._title)
            print(self.sect_hdr.format(n_title, "NTITLE"), file=psffile)
            for title in self._title:
                print(title, file=psffile)
            print(file=psffile)
            self._write_atoms(psffile)
            for section in self.sections:
                self._write_sec(psffile, section)
            self._write_other(psffile)

    def _write_atoms(self, psffile: TextIO):
        """Write atom section in a Charmm PSF file.

        Normal (standard) and extended (EXT) PSF format are
        supported.


        CHARMM Format from ``source/psffres.src``:

        no CHEQ::
         standard format:
            (I8,1X,A4,1X,A4,1X,A4,1X,A4,1X,I4,1X,2G14.6,I8)
            (I8,1X,A4,1X,A4,1X,A4,1X,A4,1X,I4,1X,2G14.6,I8,2G14.6) CHEQ
            (I8,1X,A4,1X,A4,1X,A4,1X,A4,1X,A6,1X,2G14.6,I8)  XPLOR
            (I8,1X,A4,1X,A4,1X,A4,1X,A4,1X,A6,1X,2G14.6,I8,2G14.6)  XPLOR,CHEQ
            (I8,1X,A4,1X,A4,1X,A4,1X,A4,1X,A4,1X,2G14.6,I8)  XPLOR,c35
            (I8,1X,A4,1X,A4,1X,A4,1X,A4,1X,A4,1X,2G14.6,I8,2G14.6) XPLOR,c35,CHEQ
          expanded format EXT:
            (I10,1X,A8,1X,A8,1X,A8,1X,A8,1X,I4,1X,2G14.6,I8)
            (I10,1X,A8,1X,A8,1X,A8,1X,A8,1X,I4,1X,2G14.6,I8,2G14.6) CHEQ
            (I10,1X,A8,1X,A8,1X,A8,1X,A8,1X,A6,1X,2G14.6,I8) XPLOR
            (I10,1X,A8,1X,A8,1X,A8,1X,A8,1X,A6,1X,2G14.6,I8,2G14.6) XPLOR,CHEQ
            (I10,1X,A8,1X,A8,1X,A8,1X,A8,1X,A4,1X,2G14.6,I8) XPLOR,c35
            (I10,1X,A8,1X,A8,1X,A8,1X,A8,1X,A4,1X,2G14.6,I8,2G14.6) XPLOR,c35,CHEQ
        """
        fmt: str = self._fmt[self._fmtkey]
        print(self.sect_hdr.format(self._universe.atoms.n_atoms, "NATOM"),
              file=psffile)
        atoms: mda.AtomGroup = self._universe.atoms
        lines: Tuple[np.ndarray, ...] = (np.arange(atoms.n_atoms) + 1,
                                         atoms.segids, atoms.resids,
                                         atoms.resnames, atoms.names,
                                         atoms.types, atoms.charges,
                                         atoms.masses, np.zeros_like(atoms.ids))
        lines: pd.DataFrame = pd.concat([pd.DataFrame(_) for _ in lines],
                                        axis=1)

        if self._cheq:
            fmt += "%10.6f%18s"
            cheq: Tuple[np.ndarray, ...] = (np.zeros_like(atoms.masses),
                                            np.full_like(
                                                atoms.names.astype(object),
                                                "-0.301140E-02"))
            cheq: pd.DataFrame = pd.concat([pd.DataFrame(_) for _ in cheq],
                                           axis=1)
            lines: pd.DataFrame = pd.concat([lines, cheq], axis=1)
        np.savetxt(psffile, lines, fmt=fmt)
        print(file=psffile)

    def _write_sec(self, psffile: TextIO, section_info: Tuple[str, str, int]):
        attr, header, n_perline = section_info

        if (not hasattr(self._universe, attr) or
                len(getattr(self._universe, attr).to_indices()) < 2):
            print(self.sect_hdr.format(0, header), file=psffile)
            print("\n", file=psffile)
            return

        values: np.ndarray = np.asarray(
            getattr(self._universe, attr).to_indices()) + 1
        values: np.ndarray = values.astype(object)
        n_rows, n_cols = values.shape
        n_values: int = n_perline // n_cols
        if n_rows % n_values > 0:
            n_extra: int = n_values - (n_rows % n_values)
            values: np.ndarray = np.concatenate(
                (values, np.full((n_extra, n_cols), "", dtype=np.object)))
        values: np.ndarray = values.reshape(
            (values.shape[0] // n_values, n_perline))
        print(self.sect_hdr.format(n_rows, header), file=psffile)
        np.savetxt(psffile, values, fmt="%{:d}s".format(self.col_width),
                   delimiter="")
        print(file=psffile)

    def _write_other(self, psffile: TextIO):
        n_atoms: int = self._universe.atoms.n_atoms
        n_cols: int = 8
        dn_cols: int = n_atoms % n_cols
        missing: int = n_cols - dn_cols if dn_cols > 0 else dn_cols

        # NNB
        nnb: np.ndarray = np.full(n_atoms, "0", dtype=np.object)
        if missing > 0:
            nnb: np.ndarray = np.concatenate(
                [nnb, np.full(missing, "", dtype=object)])
        nnb: np.ndarray = nnb.reshape((nnb.size // n_cols, n_cols))

        print(self.sect_hdr.format(0, "NNB\n"), file=psffile)
        np.savetxt(psffile, nnb, fmt="%{:d}s".format(self.col_width),
                   delimiter="")
        print(file=psffile)

        # NGRP NST2
        print(self.sect_hdr2.format(1, 0, "NGRP NST2"), file=psffile)
        line: np.ndarray = np.zeros(3, dtype=np.int)
        line = line.reshape((1, line.size))
        np.savetxt(psffile, line, fmt="%{:d}d".format(self.col_width),
                   delimiter="")
        print(file=psffile)

        # MOLNT
        if self._cheq:
            line: np.ndarray = np.full(n_atoms, "1", dtype=np.object)
            if dn_cols > 0:
                line: np.ndarray = np.concatenate(
                    [line, np.zeros(missing, dtype=object)])
            line: np.ndarray = line.reshape((line.size // n_cols, n_cols))
            print(self.sect_hdr.format(1, "MOLNT"), file=psffile)
            np.savetxt(psffile, line, fmt="%{:d}s".format(self.col_width),
                       delimiter="")
            print(file=psffile)
        else:
            print(self.sect_hdr.format(0, "MOLNT"), file=psffile)
            print("\n", file=psffile)

        # NUMLP NUMLPH
        print(self.sect_hdr2.format(0, 0, "NUMLP NUMLPH"), file=psffile)
        print("\n", file=psffile)

        # NCRTERM: cross-terms
        print(self.sect_hdr.format(0, "NCRTERM: cross-terms"), file=psffile)
        print("\n", file=psffile)

