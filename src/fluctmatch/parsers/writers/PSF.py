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
"""CHARMM PSF writer (for CHARMM36 style)."""

import logging
import textwrap
from pathlib import Path
from typing import ClassVar, Mapping, Optional, TextIO, Tuple, Union

import MDAnalysis as mda
import numpy as np
import static_frame as sf

from .. import base

logger: logging.Logger = logging.getLogger(__name__)


class Writer(base.TopologyWriterBase):
    """PSF writer that implements the CHARMM PSF topology format.

    Requires the following attributes to be present:
    - ids
    - names
    - core
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
        EXT_XPLOR_C35="%10d %-8s %-8d %-8s %-8s %-4s %14.6f%14.6f%8d",
    )

    def __init__(
        self,
        filename: Union[str, Path],
        *,
        extended: bool = True,
        cmap: bool = True,
        cheq: bool = True,
        charmm_version: int = 41,
        n_atoms: Optional[int] = None,
    ) -> None:
        super().__init__()

        self.filename = Path(filename).with_suffix(".psf")
        self._extended = extended
        self._cmap = cmap
        self._cheq = cheq
        self._version = charmm_version
        self._universe: Optional[mda.Universe] = None
        self._fmtkey = "EXT" if self._extended else "STD"
        self.n_atoms = n_atoms

        self.col_width = 10 if self._extended else 8
        self.sect_hdr = "{:>10d} !{}" if self._extended else "{:>8d} !{}"
        self.sect_hdr2 = "{:>10d}{:>10d} !{}" if self._extended else "{:>8d}{:>8d} !{}"
        self.sections: Tuple[Tuple[str, str, int], ...] = (
            ("bonds", "NBOND: bonds", 8),
            ("angles", "NTHETA: angles", 9),
            ("dihedrals", "NPHI: dihedrals", 8),
            ("impropers", "NIMPHI: impropers", 8),
            ("donors", "NDON: donors", 8),
            ("acceptors", "NACC: acceptors", 8),
        )

    def write(self, universe: Union[mda.Universe, mda.AtomGroup]) -> None:
        """Write universe to PSF format.

        Parameters
        ----------
        universe : :class:`~MDAnalysis.Universe` or :class:`~MDAnalysis.AtomGroup`
            A collection of atoms in a universe or atomgroup with bond
            definitions.
        """
        try:
            self._universe = universe.copy()
        except TypeError:
            self._universe = mda.Universe(
                universe.filename, universe.trajectory.filename
            )
        xplor = not np.issubdtype(universe.atoms.types.dtype, np.number)

        header = "PSF"
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

        with open(self.filename, mode="w") as psffile:
            print(header, file=psffile)
            print(file=psffile)
            n_title = len(self.title.strip().split("\n"))
            print(self.sect_hdr.format(n_title, "NTITLE"), file=psffile)
            print(textwrap.dedent(self.title).strip(), file=psffile)
            print(file=psffile)
            self._write_atoms(psffile)
            for section in self.sections:
                self._write_sec(psffile, section)
            self._write_other(psffile)

    def _write_atoms(self, psffile: TextIO, xplor: bool = True) -> None:
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
        print(
            self.sect_hdr.format(self._universe.atoms.n_atoms, "NATOM"), file=psffile,
        )
        atoms: mda.AtomGroup = self._universe.atoms
        atoms.charges[atoms.charges == -0.0] = 0.0
        types, index = np.unique(atoms.types, return_index=True)
        int_types = {k: v for k, v in zip(types, index)}
        atom_types = atoms.types if xplor else [int_types[_] for _ in atoms.types]
        lines = sf.Frame.from_concat(
            (
                sf.Series(np.arange(atoms.n_atoms) + 1),
                sf.Series(atoms.segids),
                sf.Series(atoms.resids),
                sf.Series(atoms.resnames),
                sf.Series(atoms.names),
                sf.Series(atom_types),
                sf.Series(atoms.charges),
                sf.Series(atoms.masses),
                sf.Series(np.zeros_like(atoms.ids)),
            ),
            columns="# segid resid resname name type charge mass id".split(),
            axis=1,
        )

        if self._cheq:
            fmt += "%10.6f%18s"
            cheq = sf.Frame.from_concat(
                (
                    sf.Series(np.zeros_like(atoms.masses)),
                    sf.Series(np.full_like(atoms.names, "-0.301140E-02")),
                ),
                columns="x cheq".split(),
                axis=1,
            )
            lines = sf.Frame.from_concat([lines, cheq], axis=1)
        np.savetxt(psffile, lines.values, fmt=fmt)
        print(file=psffile)

    def _write_sec(self, psffile: TextIO, section_info: Tuple[str, str, int]):
        attr, header, n_perline = section_info

        if (
            not hasattr(self._universe, attr)
            or len(getattr(self._universe, attr).to_indices()) < 2
        ):
            print(self.sect_hdr.format(0, header), file=psffile)
            print("\n", file=psffile)
            return

        values = np.asarray(getattr(self._universe, attr).to_indices()) + 1
        values = values.astype(object)
        n_rows, n_cols = values.shape
        n_values: int = n_perline // n_cols
        if n_rows % n_values > 0:
            n_extra: int = n_values - (n_rows % n_values)
            values = np.concatenate(
                (values, np.full((n_extra, n_cols), "", dtype=np.object))
            )
        values = values.reshape((values.shape[0] // n_values, n_perline))
        print(self.sect_hdr.format(n_rows, header), file=psffile)
        np.savetxt(psffile, values, fmt=f"%{self.col_width:d}s", delimiter="")
        print(file=psffile)

    def _write_other(self, psffile: TextIO):
        n_atoms = self._universe.atoms.n_atoms
        n_cols = 8
        dn_cols = n_atoms % n_cols
        missing = n_cols - dn_cols if dn_cols > 0 else dn_cols

        # NNB
        nnb = np.full(n_atoms, "0", dtype=np.object)
        if missing > 0:
            nnb = np.concatenate([nnb, np.full(missing, "", dtype=object)])
        nnb = nnb.reshape((nnb.size // n_cols, n_cols))

        print(self.sect_hdr.format(0, "NNB\n"), file=psffile)
        np.savetxt(psffile, nnb, fmt=f"%{self.col_width:d}s", delimiter="")
        print(file=psffile)

        # NGRP NST2
        print(self.sect_hdr2.format(1, 0, "NGRP NST2"), file=psffile)
        line = np.zeros(3, dtype=np.int)
        line = line.reshape((1, line.size))
        np.savetxt(psffile, line, fmt=f"%{self.col_width:d}d", delimiter="")
        print(file=psffile)

        # MOLNT
        if self._cheq:
            line = np.full(n_atoms, "1", dtype=np.object)
            if dn_cols > 0:
                line = np.concatenate([line, np.zeros(missing, dtype=object)])
            line = line.reshape((line.size // n_cols, n_cols))
            print(self.sect_hdr.format(1, "MOLNT"), file=psffile)
            np.savetxt(psffile, line, fmt=f"%{self.col_width:d}s", delimiter="")
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
