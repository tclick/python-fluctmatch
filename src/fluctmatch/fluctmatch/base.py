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
"""Defines the base class for fluctuation matching."""

import abc
from pathlib import Path
from typing import Optional, TypeVar

import static_frame as sf
from scipy import constants

TFMBase = TypeVar("TFMBase", bound="FluctMatchBase")


class FluctMatchBase(metaclass=abc.ABCMeta):
    """Base class for fluctuation matching."""

    def __init__(
        self,
        *,
        temperature: float = 300.0,
        output_dir: Optional[str] = None,
        prefix: str = "fluctmatch",
    ) -> None:
        """Initialization of fluctuation matching.

        Parameters
        ----------
        output_dir, str, optional
            Output directory
        temperature : float
            Temperature (in K)
        prefix : Optional[str]
            Filename prefix
        """
        if temperature < 0:
            raise IOError("Temperature cannot be negative.")

        self.data = dict(temperature=temperature)
        self.output_dir = Path.home() if output_dir is None else Path(output_dir)
        self.prefix = Path(prefix)

        # Attempt to create the necessary subdirectory
        self.output_dir.mkdir(exist_ok=True, parents=True)

        # Boltzmann constant (in kcal/mol)
        self.BOLTZMANN: float = temperature * (
            constants.k * constants.N_A / (constants.calorie * constants.kilo)
        )

        # Bond factor mol^2-Ang./kcal^2
        self.K_FACTOR = 0.02

    @abc.abstractmethod
    def calculate(self: TFMBase, *, target: Optional[sf.Frame] = None) -> sf.Frame:
        """Calculate the force constants from the fluctuations."""
        pass

    @abc.abstractmethod
    def initialize(
        self: TFMBase,
        *,
        input_dir: Optional[str] = None,
        executable: Optional[str] = None,
        trajectory: Optional[str] = None,
        logfile: Optional[str] = None,
    ) -> Optional[sf.Frame]:
        pass

    @abc.abstractmethod
    def simulate(
        self: TFMBase,
        *,
        input_dir: Optional[str] = None,
        executable: Optional[str] = None,
        trajectory: Optional[str] = None,
        logfile: Optional[str] = None,
    ) -> Optional[sf.Frame]:
        """Perform normal mode analysis of the system."""
        pass
