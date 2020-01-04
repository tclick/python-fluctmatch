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
import functools
import glob
import multiprocessing as mp
from os import path
from pathlib import Path
from typing import Dict
from typing import Generator
from typing import List
from typing import Union

import numpy as np
import pandas as pd
from MDAnalysis.coordinates.core import reader
from MDAnalysis.lib.util import openany

_header: List[str] = ["I", "J"]
_index: Dict[str, List[str]] = dict(
    general=["segidI", "resI", "I", "segidJ", "resJ", "J"],
    complete=["segidI", "resI", "resnI", "I", "segidJ", "resJ", "resnJ", "J"],
)


def _create_table(directory: Union[str, Path],
                  intcor: str="average.ic",
                  parmfile: str="fluctmatch.dist.prm",
                  tbltype: str="Kb",
                  verbose: bool=False) -> pd.DataFrame:
    if path.isdir(directory):
        if verbose:
            print(f"Reading directory {directory}")
        with reader(path.join(directory, intcor)) as ic_file:
            if verbose:
                print(f"    Processing {Path(directory) / intcor}...")
            ic_table: pd.DataFrame = ic_file.read()
            ic_table.set_index(_header, inplace=True)
        with reader(Path(directory) / parmfile) as prm_file:
            if verbose:
                print(f"    Processing {Path(directory) / parmfile}...")
            prm_table: pd.DataFrame = prm_file.read()["BONDS"].set_index(_header)
        table: pd.DataFrame = pd.concat([ic_table, prm_table], axis=1)
        table.reset_index(inplace=True)
        table: pd.DataFrame = table.set_index(_index["general"])[tbltype].to_frame()
        table.columns = [Path(directory).name, ]
        return table


class ParamTable(object):
    """Create a parameter table time series for distance or coupling strength.

    """

    def __init__(self,
                 prefix: str="fluctmatch",
                 tbltype: str="Kb",
                 ressep: int=3,
                 datadir: Union[str, Path]=Path.cwd()):
        """
        Parameters
        ----------
        prefix : str, optional
            Filename prefix for files
        tbltype : {"Kb", "b0"}, optional
            Table to create (coupling strength or bond distance)
        ressep : int, optional
            Number of residues to exclude from interactions.
        datadir : str, optional
            Directory with data subdirectories
        """
        self._prefix: str = prefix
        self._tbltype: str = tbltype
        self._ressep: int = ressep
        self._datadir: Union[str, Path] = datadir
        self.table: pd.DataFrame = pd.DataFrame()
        self._filenames: Dict[str, str] = dict(
            intcor="fluct.ic",
            param=".".join((self._prefix, "dist", "prm")),
        )

    def __add__(self, other: "ParamTable") -> "ParamTable":
        return self.table.add(other.table, fill_value=0.0)

    def __sub__(self, other: "ParamTable") -> "ParamTable":
        return self.table.subtract(other.table, fill_value=0.0)

    def __mul__(self, other: "ParamTable") -> "ParamTable":
        return self.table.multiply(other.table, fill_value=0.0)

    def __truediv__(self, other: "ParamTable") -> "ParamTable":
        return self.table.divide(other.table, fill_value=0.0)

    def _separate(self, prm_table: pd.DataFrame) -> pd.DataFrame:
        index = prm_table.index.names
        table: pd.DataFrame = prm_table.reset_index()
        tmp: pd.DataFrame = table[table["segidI"] == table["segidJ"]]
        tmp: pd.DataFrame = tmp[(tmp["resI"] >= tmp["resJ"] + self._ressep)
                                | (tmp["resJ"] >= tmp["resI"] + self._ressep)]
        diff: pd.DataFrame = table[table["segidI"] != table["segidJ"]]
        table: pd.DataFrame = pd.concat([tmp, diff], axis=0)
        table.set_index(index, inplace=True)
        return table

    def _complete_table(self):
        """Create a full table by reversing I and J designations.

        Returns
        -------

        """
        revcol: List[str, ...] = ["segidJ", "resJ", "J", "segidI", "resI", "I"]

        columns: np.ndarray = np.concatenate((revcol, self.table.columns[len(revcol):]))
        temp: pd.DataFrame = self.table.copy(deep=True)
        same: pd.DataFrame = temp[(temp["segidI"] == temp["segidJ"])
                                  & (temp["resI"] != temp["resJ"])]

        diff: pd.DataFrame = temp[temp["segidI"] != temp["segidJ"]]
        temp: pd.DataFrame = pd.concat([same, diff], axis=0)
        temp.columns = columns
        self.table: pd.DataFrame = pd.concat([self.table, temp], axis=0)

    def run(self, verbose: bool=False):
        """Create the time series.

        Parameters
        ----------
        verbose : bool, optional
            Print each directory as it is being processed
        """
        directories: Generator = glob.iglob(path.join(self._datadir, "*"))
        create_table = functools.partial(
            _create_table,
            intcor=self._filenames["intcor"],
            parmfile=self._filenames["param"],
            tbltype=self._tbltype,
            verbose=verbose,
        )
        pool: mp.Pool = mp.Pool()
        tables = pool.map_async(create_table, directories)
        pool.close()
        pool.join()
        tables.wait()

        self.table: pd.DataFrame = pd.concat(tables.get(), axis=1)
        self.table.columns = self.table.columns.astype(np.int)
        self.table: pd.DataFrame = self.table[np.sort(self.table.columns)]
        self.table.reset_index(inplace=True)

        self._complete_table()
        self.table.set_index(_index["general"], inplace=True)
        self.table.fillna(0., inplace=True)
        self.table.sort_index(kind="mergesort", inplace=True)

    def from_file(self, filename: Union[str, Path]):
        """Load a parameter table from a file.

        Parameters
        ----------
        filename : str or stream
            Filename of the parameter table.
        """
        with open(filename, mode="r") as table:
            self.table = pd.read_csv(table.name, skipinitialspace=True,
                                     delim_whitespace=True, header=0)
            if "resnI" in self.table.columns:
                self.table.set_index(_index["complete"], inplace=True)
            else:
                self.table.set_index(_index["general"], inplace=True)
            self.table.columns = self.table.columns.astype(np.int)

    def write(self, filename: Union[str, Path]):
        """Write the parameter table to file.

        Parameters
        ----------
        filename : str or stream
            Location to write the parameter table.
        """
        with openany(filename, mode="w") as table:
            self.table.to_csv(table, header=True, index=True,
                              float_format="%.6f", encoding="utf-8")

    @property
    def per_residue(self) -> pd.DataFrame:
        """Create a single residue time series.

        Returns
        -------
        A table with values per residue.
        """
        # Separate by residue
        table: pd.DataFrame = self._separate(self.table)
        table: pd.DataFrame = 0.5 * table.groupby(level=["segidI", "resI"]).sum()
        table.sort_index(axis=1, inplace=True)
        table: pd.DataFrame = table.reindex(index=table.index,
                                            columns=np.sort(table.columns))
        return table

    @property
    def interactions(self) -> pd.DataFrame:
        """Create a time series for the residue-residue interactions.

        Returns
        -------
        A table with residue-residue values.
        """
        table: pd.DataFrame = self._separate(self.table)
        table: pd.DataFrame = table.groupby(level=["segidI", "resI",
                                                   "segidJ", "resJ"]).sum()
        table.sort_index(axis=1, inplace=True)
        table: pd.DataFrame = table.reindex(index=table.index,
                                            columns=np.sort(table.columns))
        return table
