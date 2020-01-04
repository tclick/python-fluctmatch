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

import abc
import os
from pathlib import Path
from typing import Dict
from typing import List
from typing import NoReturn


class FluctMatchBase(metaclass=abc.ABCMeta):
    """Base class for fluctuation matching."""

    def __init__(self, *args: List, **kwargs: Dict):
        """Initialization of fluctuation matching.

        Parameters
        ----------
        topology : filename or Topology object
            A CHARMM/XPLOR PSF topology file, PDB file or Gromacs GRO file; used
            to define the list of atoms. If the file includes bond information,
            partial charges, atom masses, ... then these data will be available
            to MDAnalysis. A "structure" file (PSF, PDB or GRO, in the sense of
            a topology) is always required. Alternatively, an existing
            :class:`MDAnalysis.core.topology.Topology` instance may also be
            given.
        extended
            Renames the residues and atoms according to the extended CHARMM PSF
            format. Standard CHARMM PSF limits the residue and atom names to
            four characters, but the extended CHARMM PSF permits eight
            characters. The residues and atoms are renamed according to the
            number of segments (1: A, 2: B, etc.) and then the residue number or
            atom index number.
         xplor
            Assigns the atom type as either a numerical or an alphanumerical
            designation. CHARMM normally assigns a numerical designation, but
            the XPLOR version permits an alphanumerical designation with a
            maximum size of 4. The numerical form corresponds to the atom index
            number plus a factor of 100, and the alphanumerical form will be
            similar the standard CHARMM atom name.
        topology_format
            Provide the file format of the topology file; ``None`` guesses it
            from the file extension [``None``] Can also pass a subclass of
            :class:`MDAnalysis.topology.base.TopologyReaderBase` to define a
            custom reader to be used on the topology file.
        format
            Provide the file format of the coordinate or trajectory file;
            ``None`` guesses it from the file extension. Note that this keyword
            has no effect if a list of file names is supplied because the
            "chained" reader has to guess the file format for each individual
            list member. [``None``] Can also pass a subclass of
            :class:`MDAnalysis.coordinates.base.ProtoReader` to define a custom
            reader to be used on the trajectory file.
        guess_bonds : bool, optional
            Once Universe has been loaded, attempt to guess the connectivity
            between atoms.  This will populate the .bonds .angles and .dihedrals
            attributes of the Universe.
        vdwradii : dict, optional
            For use with *guess_bonds*. Supply a dict giving a vdwradii for each
            atom type which are used in guessing bonds.
        is_anchor : bool, optional
            When unpickling instances of
            :class:`MDAnalysis.core.groups.AtomGroup` existing Universes are
            searched for one where to anchor those atoms. Set to ``False`` to
            prevent this Universe from being considered. [``True``]
        anchor_name : str, optional
            Setting to other than ``None`` will cause
            :class:`MDAnalysis.core.groups.AtomGroup` instances pickled from the
            Universe to only unpickle if a compatible Universe with matching
            *anchor_name* is found. Even if *anchor_name* is set *is_anchor*
            will still be honored when unpickling.
        in_memory
            After reading in the trajectory, transfer it to an in-memory
            representations, which allow for manipulation of coordinates.
        in_memory_step
            Only read every nth frame into in-memory representation.
        outdir
            Output directory
        temperature
            Temperature (in K)
        rmin
            Minimum distance to consider for bond lengths.
        rmax
            Maximum distance to consider for bond lengths.
        """
        self.parameters: Dict = dict()
        self.target: Dict = dict()

        self.outdir: Path = Path(kwargs.get("outdir", os.getcwd()))
        self.prefix: Path = Path(kwargs.get("prefix", "fluctmatch"))
        self.temperature: float = kwargs.get("temperature", 300.0)
        if self.temperature < 0:
            raise IOError("Temperature cannot be negative.")
        self.args: List = args
        self.kwargs: Dict = kwargs

        # Attempt to create the necessary subdirectory
        self.outdir.mkdir(exist_ok=True, parents=True)

    @abc.abstractmethod
    def initialize(self, nma_exec: str = None, restart: bool = False) -> NoReturn:
        """Create an elastic network model from a basic coarse-grain model.

        Parameters
        ----------
        nma_exec : str
            executable file for normal mode analysis
        restart : bool, optional
            Reinitialize the object by reading files instead of doing initial
            calculations.
        """
        pass

    @abc.abstractmethod
    def run(
        self,
        nma_exec: str = None,
        tol: float = 1.0e-4,
        min_cycles: int = 200,
        max_cycles: int = 200,
        force_tol: float = 0.02,
    ) -> NoReturn:
        """Perform a self-consistent fluctuation matching.

        Parameters
        ----------
        nma_exec : str
            executable file for normal mode analysis
        tol : float, optional
            error tolerance
        min_cycles : int, optional
            minimum number of fluctuation matching cycles
        max_cycles : int, optional
            maximum number of fluctuation matching cycles
        force_tol : float, optional
            force constants <= force tolerance become zero after min_cycles
        """
        pass
