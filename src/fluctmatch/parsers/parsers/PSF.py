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
"""CHARMM PSF reader (for CHARMM36 style) and writer."""

import logging
from typing import Callable, ClassVar, List, Mapping, TextIO, Tuple, Union

import numpy as np
from MDAnalysis.core.topology import Topology
from MDAnalysis.core.topologyattrs import (
    Angles,
    Atomids,
    Atomnames,
    Atomtypes,
    Bonds,
    Charges,
    Dihedrals,
    Impropers,
    Masses,
    Resids,
    Resnames,
    Resnums,
    Segids,
)
from MDAnalysis.lib.util import FORTRANReader
from MDAnalysis.topology import PSFParser
from MDAnalysis.topology.base import change_squash

logger: logging.Logger = logging.getLogger(__name__)


# Changed the segid squash_by to change_squash to prevent segment ID sorting.
class Reader(PSFParser.PSFParser):
    """Read topology information from a CHARMM/NAMD/XPLOR PSF_ file.

    Creates a Topology with the following Attributes:
    - ids
    - names
    - core
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

    def parse(self, **kwargs: Mapping) -> Topology:
        """Parse PSF file into Topology

        Returns
        -------
        MDAnalysis *Topology* object
        """
        # Open and check psf validity
        with open(self.filename) as psffile:
            header: str = next(psffile)
            if not header.startswith("PSF"):
                err: str = (
                    f"{self.filename} is not valid PSF file (header = {header})"
                )
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
            top: Topology = self._parse_sec(psffile, ("NATOM", 1, 1, self._parseatoms))
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

    def _parseatoms(
        self, lines: Callable[[TextIO], str], atoms_per: int, numlines: int
    ) -> Topology:
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
        the CHARMM "EXTended" PSF format cannot accomodate long atom core, and
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
                    atom_parser: FORTRANReader = FORTRANReader(atom_parsers["NAMD"])
                    vals: List[str] = atom_parser.read(line)
                    logger.warning(
                        "Guessing that this is actually a NAMD-type "
                        "PSF file... continuing with fingers "
                        "crossed!"
                    )
                    logger.info(f"First NAMD-type line: {i}: {line.rstrip()}")
                except ValueError:
                    atom_parser: FORTRANReader = FORTRANReader(
                        atom_parsers[self._format].replace("A6", "A4")
                    )
                    vals: List[str] = atom_parser.read(line)
                    logger.warning(
                        "Guessing that this is actually a pre "
                        "CHARMM36 PSF file... continuing with "
                        "fingers crossed!"
                    )
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
        charges[charges == -0.0] = 0.0
        charges: Charges = Charges(charges)
        masses: Masses = Masses(masses)

        # Residue
        # resids, resnames
        residx, (new_resids, new_resnames, perres_segids) = change_squash(
            (resids, resnames, segids), (resids, resnames, segids)
        )
        # transform from atom:Rid to atom:Rix
        residueids: Resids = Resids(new_resids)
        residuenums: Resnums = Resnums(new_resids.copy())
        residuenames: Resnames = Resnames(new_resnames)

        # Segment
        segidx, (perseg_segids,) = change_squash((perres_segids,), (perres_segids,))
        segids: Segids = Segids(perseg_segids)

        top: Topology = Topology(
            len(atomids),
            len(new_resids),
            len(segids),
            attrs=[
                atomids,
                atomnames,
                atomtypes,
                charges,
                masses,
                residueids,
                residuenums,
                residuenames,
                segids,
            ],
            atom_resindex=residx,
            residue_segindex=segidx,
        )

        return top
