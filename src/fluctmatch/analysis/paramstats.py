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

import numpy as np
import pandas as pd


class ParamStats:
    """Calculate parameter statistics from a parameter table.
    """

    def __init__(self, table: pd.DataFrame):
        """

        Parameters
        ----------
        table : :class:`ParamTable`
        """
        self._table: pd.DataFrame = table

    def table_stats(self) -> pd.DataFrame:
        """Calculate several statistics about the table.

        Returns
        -------
        A table of statistics for the overall table.
        """
        info: pd.DataFrame = self._table.table.T.describe().T
        return info.drop("count", axis=1)

    def table_hist(self) -> pd.Series:
        """Calculate a histogram of the overall table.

        Returns
        -------
        A `pandas.Series` histogram
        """
        hist, bin_edges = np.histogram(self._table.table, bins=100, density=True)
        edges: np.ndarray = (bin_edges[1:] + bin_edges[:-1]) / 2
        return pd.Series(hist, index=edges)

    def interaction_stats(self) -> pd.DataFrame:
        """Calculate statistics for the residue-residue interactions.

        Returns
        -------
        A table of statistics for the residue-residue interactions
        """
        info: pd.DataFrame = self._table.interactions.T.describe().T
        return info.drop("count", axis=1)

    def interaction_hist(self) -> pd.Series:
        """Calculate a histogram of the residue-residue interactions.

        Returns
        -------
        A `pandas.Series` histogram
        """
        hist, bin_edges = np.histogram(
            self._table.interactions, bins="auto", density=True
        )
        edges: np.ndarray = (bin_edges[1:] + bin_edges[:-1]) / 2
        return pd.Series(hist, index=edges)

    def residue_stats(self) -> pd.DataFrame:
        """Calculate statistics for the individual residues.

        Returns
        -------
        A table of statistics for the residue-residue interactions
        """
        info: pd.DataFrame = self._table.per_residue.T.describe().T
        return info.drop("count", axis=1)

    def residue_hist(self) -> pd.Series:
        """Calculate a histogram of the individual residues.

        Returns
        -------
        A `pandas.Series` histogram
        """
        hist, bin_edges = np.histogram(
            self._table.per_residue, bins="auto", density=True
        )
        edges: np.ndarray = (bin_edges[1:] + bin_edges[:-1]) / 2
        return pd.Series(hist, index=edges)
