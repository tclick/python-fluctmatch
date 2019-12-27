# fluctmatch --- https://github.com/tclick/python-fluctmatch
# Copyright (c) 2013-2017 The fluctmatch Development Team and contributors
# (see the file AUTHORS for the full list of names)
#
# Released under the New BSD license.
#
# Please cite your use of fluctmatch in published work:
#
# Timothy H. Click, Nixon Raj, and Jhih-Wei Chu.
# Calculation of Enzyme Fluctuograms from All-Atom Molecular Dynamics
# Simulation. Meth Enzymology. 578 (2016), 327-342,
# doi:10.1016/bs.mie.2016.05.024.
#
import numpy as np
import pandas as pd


class ParamStats(object):
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
        hist, bin_edges = np.histogram(
            self._table.table, bins=100, density=True)
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
            self._table.interactions, bins="auto", density=True)
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
            self._table.per_residue, bins="auto", density=True)
        edges: np.ndarray = (bin_edges[1:] + bin_edges[:-1]) / 2
        return pd.Series(hist, index=edges)
