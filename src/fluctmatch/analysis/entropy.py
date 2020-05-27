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

from typing import List

import numpy as np
import pandas as pd
from scipy import stats

from fluctmatch.analysis.paramtable import ParamTable


class Entropy(object):
    """Calculate various entropic contributions from the coupling strengths.
    """

    def __init__(self, filename, ressep: int = 3):
        """
        Parameters
        ----------
        filename : str
            Name of the file with a complete coupling strength time series
        ressep : int, optional
            Residue separation
        """
        self._table: pd.DataFrame = ParamTable(ressep=ressep)
        self._table.from_file(filename=filename)

    def relative_entropy(self) -> pd.DataFrame:
        """Calculate the relative entropy between windows.

        Calculate the relative entropy of a coarse-grain system by using the
        following equations:

        .. math::
            P^t_{IJ} = K^t_{IJ} / \sum {K^t_{IJ}}
            S = \sum {P^t_{IJ} ln(P^t_{IJ}}

        where :math:`t/2` is the previous window because of the overlap in
        trajectory times by :math:`t/2`. If :math:`Q^{t/2}_{IJ} = 0.0`, then
        it is replaced with a penalty value determined by a force constant value
        as determined by the highest probability of nonzero force constants
        within the overall time series.

        The required table must be a time series of an internal coordinates
        table containing bond force constants.

        Returns
        -------
        Timeseries of relative entropy per residue
        """
        header: List[str, str] = ["segidI", "resI"]

        entropy: pd.DataFrame = self._table._separate(self._table.table)
        entropy: pd.DataFrame = entropy.groupby(level=header).apply(
            lambda x: x / x.sum()
        )
        entropy: pd.DataFrame = entropy.groupby(level=header).agg(stats.entropy)
        entropy.replace(-np.inf, 0.0, inplace=True)

        return entropy

    def coupling_entropy(self) -> pd.DataFrame:
        """Calculate the entropic contributions between residues.

        Returns
        -------
        Entropy of residue-residue interactions over time
        """
        # Transpose matrix because stats.entropy does row-wise calculation.
        table: pd.DataFrame = self._table.per_residue
        entropy: pd.DataFrame = stats.entropy(table.T)
        return pd.DataFrame(entropy, index=table.index)

    def windiff_entropy(self, bins: int = 100) -> pd.DataFrame:
        """Calculate the relative entropy between windows.

        Calculate the relative entropy of a coarse-grain system by using the
        following equations:

        .. math::

            P^t_{IJ} = K^t_{IJ} / \leftangle K^t_{IJ}\rightangle
            Q^t/2_{IJ} = K^{t/2}_{IJ} / \leftangle K^{t/2}_{IJ}
            S = -0.5 * (\sum(P^t_{IJ}\ln(P^t_{IJ} / Q^{t/2}_{IJ})) +
                \sum(Q^{t/2}_{IJ})\ln(P^t_{IJ} / Q^{t/2}_{IJ})))

        where :math:`t/2` is the previous window because of the overlap in
        trajectory times by :math:`t/2`. If :math:`Q^{t/2}_{IJ} = 0.0`, then it
        is replaced with a penalty value determined by a force constant value as
        determined by the highest probability of nonzero force constants within
        the overall time series.

        The required table must be a time series of an internal coordinates
        table containing bond force constants.

        Parameters
        ----------
        bins : int, optional
            Number of bins for histogram determination

        Returns
        -------
        Entropy difference between two windows
        """

        # Calculate value of maximum probability and define penalty value.
        def normalize(x: pd.DataFrame) -> pd.DataFrame:
            return x / x.sum()

        header: List[str, str] = ["segidI", "resI"]
        table: pd.DataFrame = self._table._separate(self._table.table)
        hist, edges = np.histogram(table, range=(1e-4, table.values.max()), bins=bins)
        hist: np.ndarray = (hist / table.size).astype(dtype=np.float)
        xaxis: np.ndarray = (edges[:-1] + edges[1:]) / 2
        try:
            penalty: np.ndarray = xaxis[np.where(hist == hist.max())][0]
        except IndexError:
            penalty: np.ndarray = xaxis[-1]

        # Calculate average coupling strength per residue.
        table[table == 0.0]: pd.DataFrame = penalty
        #     meanI = tmp.groupby(level=["resI"]).mean()
        table: pd.DataFrame = table.groupby(level=header).transform(normalize)

        # Utilize the numpy arrays for calculations
        P: pd.DataFrame = table[table.columns[1:]]
        Q: pd.DataFrame = table[table.columns[:-1]]
        P.columns = Q.columns

        # Caclulate the relative entropy
        S_P: pd.DataFrame = P * np.log(P / Q)
        S_Q: pd.DataFrame = Q * np.log(P / Q)
        S_P.fillna(0.0, inplace=True)
        S_Q.fillna(0.0, inplace=True)
        entropy: pd.DataFrame = -(
            S_P.groupby(level=header).sum() + S_Q.groupby(level=header).sum()
        )
        entropy[entropy == -0.0] = entropy[entropy == -0.0].abs()

        return entropy
