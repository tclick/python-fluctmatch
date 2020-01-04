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

from pathlib import Path

from pkg_resources import resource_filename

__all__ = [
    "PDB",  # PDB
    "GRO",
    "PDB_dna",
    "TIP3P",
    "TIP4P",
    "IONS",
    "DMA",
    "TPR",
    "XTC",  # Gromacs
    "NCSC",
    "PSF",
    "DCD",
    "IC",
    "COR",
    "PRM",
    "RTF",
    "STR"
]

PDB = resource_filename(__name__,
                        Path().joinpath("data", "trex1.pdb").as_posix())
GRO = resource_filename(__name__,
                        Path().joinpath("data", "trex2.gro").as_posix())
PDB_dna = resource_filename(__name__,
                            Path().joinpath("data", "dna.pdb").as_posix())
TPR = resource_filename(__name__,
                        Path().joinpath("data", "trex1.tpr").as_posix())
XTC = resource_filename(__name__,
                        Path().joinpath("data", "trex1.xtc").as_posix())
TIP3P = resource_filename(__name__,
                          Path().joinpath("data", "spc216.gro").as_posix())
TIP4P = resource_filename(__name__,
                          Path().joinpath("data", "tip4p.gro").as_posix())
IONS = resource_filename(__name__,
                         Path().joinpath("data", "ions.pdb").as_posix())
DMA = resource_filename(__name__,
                        Path().joinpath("data", "dma.gro").as_posix())
NCSC = resource_filename(__name__,
                         Path().joinpath("data", "ncsc.pdb").as_posix())
PSF = resource_filename(__name__,
                        Path().joinpath("data", "cg.xplor.psf").as_posix())
DCD = resource_filename(__name__,
                        Path().joinpath("data", "cg.dcd").as_posix())
IC = resource_filename(__name__,
                       Path().joinpath("data", "fluct.ic").as_posix())
PRM = resource_filename(__name__,
                        Path().joinpath("data", "fluctmatch.prm").as_posix())
COR = resource_filename(__name__,
                        Path().joinpath("data", "cg.cor").as_posix())
RTF = resource_filename(__name__,
                        Path().joinpath("data", "cg.rtf").as_posix())
STR = resource_filename(__name__,
                        Path().joinpath("data", "cg.stream").as_posix())
