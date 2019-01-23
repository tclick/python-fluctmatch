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

from collections import defaultdict
from typing import ClassVar, Iterable, List, Mapping, Union

import numpy as np
from MDAnalysis.core.groups import Atom, Residue, Segment, ComponentBase
from MDAnalysis.core.topologyattrs import Atomtypes


class XplorTypes(Atomtypes):
    """String types for atoms used for XPLOR-PSF."""
    attriname: ClassVar[str] = "xplortypes"
    singular: ClassVar[str] = "xplortype"
    target_classes: ClassVar[List[ComponentBase]] = [Atom, Residue, Segment]
    per_object: ClassVar[str] = "atom"
    transplants: ClassVar[Mapping[List]] = defaultdict(list)

    def __init__(self, values: Union[Iterable, np.ndarray],
                 guessed: bool=False):
        super().__init__(values, guessed)


class NumTypes(Atomtypes):
    """Number types for atoms used for PSF."""
    attriname: ClassVar[str] = "numtypes"
    singular: ClassVar[str] = "numtype"
    target_classes: ClassVar[List[ComponentBase]] = [Atom, Residue, Segment]
    per_object: ClassVar[str] = "atom"
    transplants: ClassVar[Mapping[List]] = defaultdict(list)

    def __init__(self, values: Union[Iterable, np.ndarray],
                 guessed: bool=False):
        super().__init__(values, guessed)
