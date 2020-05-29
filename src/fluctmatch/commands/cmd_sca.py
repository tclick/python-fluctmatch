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

import logging
import logging.config
import os
import pickle
import sys
from pathlib import Path

import click
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from fluctmatch.decomposition import ica

from ..analysis import fluctsca
from ..analysis.paramtable import ParamTable


@click.command(
    "sca", short_help="Statistical coupling analysis (SCA) on coupling strength"
)
@click.option(
    "-l",
    "--logfile",
    metavar="LOG",
    show_default=True,
    default=Path.cwd() / "fluctsca.log",
    type=click.Path(exists=False, file_okay=True, resolve_path=True),
    help="Log file",
)
@click.option(
    "-n",
    "--ntrials",
    metavar="NTRIALS",
    default=100,
    show_default=True,
    type=click.IntRange(0, None, clamp=True),
    help="Number of random iterations",
)
@click.option(
    "--std",
    metavar="STDDEV",
    default=2,
    show_default=True,
    type=click.IntRange(0, None, clamp=True),
    help="Number of std. deviations for beyond second eigenmode",
)
@click.option(
    "-k",
    "--kpos",
    metavar="KPOS",
    default=0,
    type=click.IntRange(0, None, clamp=True),
    help="Number of eigenmodes [default: auto]",
)
@click.option(
    "-p",
    "--pcut",
    default=0.95,
    show_default=True,
    type=np.float,
    help="Cutoff value for sector selection",
)
@click.option(
    "-r",
    "--ressep",
    metavar="RESSEP",
    default=3,
    show_default=True,
    type=click.IntRange(0, None, clamp=True),
    help="Number of residues to exclude in I,I+r",
)
@click.option(
    "-o",
    "--output",
    default=Path.cwd() / "scafluct.db",
    show_default=True,
    type=click.Path(exists=False, file_okay=True, resolve_path=True),
    help="Output filename",
)
@click.option(
    "-s",
    "--subset",
    metavar="SEGID RES RES",
    type=(
        str,
        click.IntRange(1, None, clamp=True),
        click.IntRange(1, None, clamp=True),
    ),
    multiple=True,
    help="Subset of a system (SEGID FIRST LAST)",
)
@click.option(
    "--all",
    "transformation",
    flag_value="all",
    default=True,
    help="Include all interactions [default]",
)
@click.option(
    "--bb",
    "transformation",
    flag_value="backbone",
    help="Include backbone-backone interactions only",
)
@click.option(
    "--sc",
    "transformation",
    flag_value="sidechain",
    help="Include sidechain-sidechain interactions only",
)
@click.option(
    "--bbsc",
    "transformation",
    flag_value="bbsc",
    help="Include backbone-sidechain interactions only",
)
@click.option(
    "-m",
    "--ictype",
    type=click.Choice(["fastica", "infomax", "extended-infomax"]),
    show_default=True,
    default="infomax",
    help="ICA method",
)
@click.argument(
    "filename", type=click.Path(exists=True, file_okay=True, resolve_path=True)
)
def cli(
    logfile,
    ntrials,
    std,
    kpos,
    pcut,
    ressep,
    output,
    subset,
    transformation,
    ictype,
    filename,
):
    # Setup logger
    logging.config.dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,  # this fixes the problem
            "formatters": {
                "standard": {
                    "class": "logging.Formatter",
                    "format": "%(name)-12s %(levelname)-8s %(message)s",
                },
                "detailed": {
                    "class": "logging.Formatter",
                    "format": "%(asctime)s %(name)-15s %(levelname)-8s %(message)s",
                    "datefmt": "%m-%d-%y %H:%M",
                },
            },
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "level": "INFO",
                    "formatter": "standard",
                },
                "file": {
                    "class": "logging.FileHandler",
                    "filename": logfile,
                    "level": "INFO",
                    "mode": "w",
                    "formatter": "detailed",
                },
            },
            "root": {"level": "INFO", "handlers": ["console", "file"]},
        }
    )
    logger: logging.Logger = logging.getLogger(__name__)

    # Load the table, separate by I,I+r, and if requested, create a subset.
    logger.info("Loading parameter table %s", click.format_filename(filename))
    table = ParamTable(ressep=ressep)
    table.from_file(click.format_filename(filename))
    kb = table._separate(table.table)
    _index = kb.index.names
    if transformation == "backbone":
        logger.info("Calculating backbone-backbone interactions.")
        kb.reset_index(inplace=True)
        kb = kb[(kb["I"] == "N") | (kb["I"] == "CA") | (kb["I"] == "O")]
        kb = kb[(kb["J"] == "N") | (kb["J"] == "CA") | (kb["J"] == "O")]
        table.table = kb.set_index(_index)
    elif transformation == "sidechain":
        logger.info("Calculating sidechain-sidechain interactions.")
        kb.reset_index(inplace=True)
        kb = kb[(kb["I"] == "CB") & (kb["J"] == "CB")]
        table.table = kb.set_index(_index)
    elif transformation == "bbsc":
        logger.info("Calculating backbone-sidechain interactions.")
        kb.reset_index(inplace=True)
        tmp1 = kb[(kb["I"] == "CB") & ((kb["J"] == "N") | (kb["J"] == "O"))]
        tmp2 = kb[(kb["J"] == "CB") & ((kb["J"] == "N") | (kb["J"] == "O"))]
        table.table = pd.concat([tmp1, tmp2], axis=0).set_index(_index)
    else:
        logger.info("Accounting for all interactions.")
    kb = table.per_residue

    D_info = dict(kb=kb, npos=kb.index.size, ressep=ressep)

    if subset:
        segid, start, stop = subset[0]
        logger.info(
            "Using a subset of {} between {:d} and {:d}".format(segid, start, stop)
        )
        kb = kb.loc[segid].loc[start:stop]
        D_info["kb"] = kb.copy(deep=True)
        D_info["subset"] = subset[0]
        D_info["Npos"] = kb.index.size

    kb: np.ndarray = kb.T.values
    U: np.ndarray = PCA(whiten=True, svd_solver="full").fit_transform(kb)
    if kpos < 1:
        logger.info("Using {:d} random trials.".format(ntrials))
        Lrand: np.ndarray = fluctsca.randomize(kb, n_trials=ntrials)

    Csca: np.ndarray = fluctsca.get_correlation(kb)
    D_sca = dict(U=U, Csca=Csca, Lrand=Lrand if kpos < 1 else None, ntrials=ntrials)

    # Determine the number of eigenmodes if kpos = 0
    Lsca, Vsca = fluctsca.eigenVect(Csca)
    _kpos: int = fluctsca.chooseKpos(Lsca, Lrand, stddev=std) if kpos == 0 else kpos
    logger.info("Selecting {:d} eigenmodes".format(_kpos))

    # Calculate IC sectors
    logger.info("Calculating the ICA for the residues.")
    time_info = ica.ICA(
        n_components=_kpos, method=ictype, max_iter=ntrials, whiten=False
    )
    try:
        Vpica: np.ndarray = time_info.fit_transform(Vsca[:, :_kpos])
    except IndexError:
        logger.error("An error occurred while using %s. Exiting...",)
        sys.exit(os.EX_DATAERR)

    ics, icsize, sortedpos, cutoff, scaled_pd, pdf = fluctsca.icList(
        Vpica, _kpos, Csca, p_cut=pcut
    )
    percentage: float = len(sortedpos) / D_info["npos"] * 100
    logger.info(
        "%d residues are within %d sectors: percentage:.2f%%",
        len(sortedpos),
        _kpos,
        percentage,
    )

    logger.info("Calculating the ICA for the windows.")
    Usca: np.ndarray = U.dot(Vsca[:, :_kpos]).dot(np.diag(1 / np.sqrt(Lsca[:_kpos])))
    Upica: np.ndarray = time_info.mixing_.dot(Usca.T).T
    for k in range(Upica.shape[1]):
        Upica[:, k] /= np.sqrt(Upica[:, k].T.dot(Upica[:, k]))

    res_info = ica.ICA(
        n_components=_kpos, method=ictype, max_iter=ntrials, whiten=False
    )
    Usica: np.ndarray = res_info.fit_transform(Usca)

    D_sector = dict(
        kpos=_kpos,
        Vsca=Vsca,
        Lsca=Lsca,
        Vpica=Vpica,
        Wpica=time_info.mixing_,
        ics=ics,
        icsize=icsize,
        sortedpos=sortedpos,
        cutoff=cutoff,
        scaled_pd=scaled_pd,
        pdf=pdf,
        Usca=Usca,
        Upica=Upica,
        Usica=Usica,
        Wsica=res_info.mixing_,
    )

    D = dict(info=D_info, sca=D_sca, sector=D_sector)
    with open(click.format_filename(output), mode="wb") as dbf:
        logger.info("Saving data to %s", click.format_filename(output))
        pickle.dump(D, dbf, protocol=pickle.HIGHEST_PROTOCOL)
