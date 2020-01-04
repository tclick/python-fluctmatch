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
import importlib
import logging
import pkgutil
from typing import MutableMapping

import MDAnalysis as mda

import fluctmatch.core.models
import fluctmatch.parsers.parsers
import fluctmatch.parsers.readers
import fluctmatch.parsers.writers

logger: logging.Logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

__version__: str = "4.0.0"


def iter_namespace(ns_pkg):
    """Iterate over a namespace package.

    Parameters
    ----------
    ns_pkg : namespace

    References
    ----------
    .. [1] https://packaging.python.org/guides/creating-and-discovering-plugins/
    """
    # Specifying the second argument (prefix) to iter_modules makes the
    # returned name an absolute name instead of a relative one. This allows
    # import_module to work without having to do additional modification to
    # the name.
    return pkgutil.iter_modules(ns_pkg.__path__, ns_pkg.__name__ + ".")


# Update the parsers in MDAnalysis
mda._PARSERS.update({
    name.split(".")[-1].upper(): importlib.import_module(name).Reader
    for _, name, _
    in iter_namespace(fluctmatch.parsers.parsers)
})
mda._PARSERS["COR"] = mda._PARSERS["CRD"]

# Update the readers in MDAnalysis
mda._READERS.update({
    name.split(".")[-1].upper(): importlib.import_module(name).Reader
    for _, name, _
    in iter_namespace(fluctmatch.parsers.readers)
})

# Update the writers in MDAnalysis
mda._SINGLEFRAME_WRITERS.update({
    name.split(".")[-1].upper(): importlib.import_module(name).Writer
    for _, name, _
    in iter_namespace(fluctmatch.parsers.writers)
})

_MODELS: MutableMapping = {
    name.split(".")[-1].upper(): importlib.import_module(name).Model
    for _, name, _
    in iter_namespace(fluctmatch.core.models)
}
