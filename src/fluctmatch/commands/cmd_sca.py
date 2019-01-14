# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding: utf-8 -*-
# vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4
#
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
import logging
import logging.config
import pickle
from pathlib import Path

import click
import numpy as np
import pandas as pd
from ..analysis.paramtable import ParamTable
from ..analysis import fluctsca
from ..fluctsca.fluctsca import FluctSCA


@click.command(
    "sca",
    short_help="Statistical coupling analysis (SCA) on coupling strength")
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
    help="Number of random iterations")
@click.option(
    "--std",
    metavar="STDDEV",
    default=2,
    show_default=True,
    type=click.IntRange(0, None, clamp=True),
    help="Number of std. deviations for beyond second eigenmode")
@click.option(
    "-k",
    "--kpos",
    metavar="KPOS",
    default=0,
    type=click.IntRange(0, None, clamp=True),
    help="Number of eigenmodes [default: auto]")
@click.option(
    "-p",
    "--pcut",
    default=0.95,
    show_default=True,
    type=np.float,
    help="Cutoff value for sector selection")
@click.option(
    "-r",
    "--ressep",
    metavar="RESSEP",
    default=3,
    show_default=True,
    type=click.IntRange(0, None, clamp=True),
    help="Number of residues to exclude in I,I+r")
@click.option(
    "-o",
    "--output",
    default=Path.cwd() / "scafluct.db",
    show_default=True,
    type=click.Path(
        exists=False,
        file_okay=True,
        resolve_path=True,
    ),
    help="Output filename")
@click.option(
    "-s",
    "--subset",
    metavar="SEGID RES RES",
    type=(str, click.IntRange(1, None, clamp=True),
          click.IntRange(1, None, clamp=True)),
    multiple=True,
    help="Subset of a system (SEGID FIRST LAST)")
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
@click.option(
    "-w",
    "--whiten",
    is_flag=True,
    help="Standardize data",
)
@click.option(
    "-t",
    "--tol",
    default=0.99,
    show_default=True,
    type=np.float,
    help="Tolerance level, if data unstandardized",
)
@click.argument(
    "filename",
    type=click.Path(
        exists=True,
        file_okay=True,
        resolve_path=True,
    ))
def cli(logfile, ntrials, std, kpos, pcut, ressep, output, subset,
        transformation, ictype, whiten, tol, filename):
    # Setup logger
    logging.config.dictConfig({
        "version": 1,
        "disable_existing_loggers": False,  # this fixes the problem
        "formatters": {
            "standard": {
                "class": "logging.Formatter",
                "format": "%(name)-12s %(levelname)-8s %(message)s",
            },
            "detailed": {
                "class": "logging.Formatter",
                "format":
                "%(asctime)s %(name)-15s %(levelname)-8s %(message)s",
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
            }
        },
        "root": {
            "level": "INFO",
            "handlers": ["console", "file"]
        },
    })
    logger = logging.getLogger(__name__)

    # Load the table, separate by I,I+r, and if requested, create a subset.
    logger.info("Loading parameter table {}".format(
        click.format_filename(filename)))
    table = ParamTable(ressep=ressep)
    table.from_file(click.format_filename(filename))
    kb: pd.DataFrame = table._separate(table.table)
    _index = kb.index.names
    if transformation == "backbone":
        logger.info("Calculating backbone-backbone interactions.")
        kb.reset_index(inplace=True)
        kb: pd.DataFrame = kb[(kb["I"] == "N") | (kb["I"] == "CA") | (kb["I"] == "O")]
        kb: pd.DataFrame = kb[(kb["J"] == "N") | (kb["J"] == "CA") | (kb["J"] == "O")]
        table.table: pd.DataFrame = kb.set_index(_index)
    elif transformation == "sidechain":
        logger.info("Calculating sidechain-sidechain interactions.")
        kb.reset_index(inplace=True)
        kb: pd.DataFrame = kb[(kb["I"] == "CB") & (kb["J"] == "CB")]
        table.table: pd.DataFrame = kb.set_index(_index)
    elif transformation == "bbsc":
        logger.info("Calculating backbone-sidechain interactions.")
        kb.reset_index(inplace=True)
        tmp1: pd.DataFrame = kb[(kb["I"] == "CB") & ((kb["J"] == "N") | (kb["J"] == "O"))]
        tmp2: pd.DataFrame = kb[(kb["J"] == "CB") & ((kb["J"] == "N") | (kb["J"] == "O"))]
        table.table: pd.DataFrame = pd.concat([tmp1, tmp2], axis=0).set_index(_index)
    else:
        logger.info("Accounting for all interactions.")
    kb: pd.DataFrame = table.per_residue

    D_info = dict(
        kb=kb.values,
        residues=kb.index.values,
        windows=kb.columns.values,
        npos=kb.index.size,
        ressep=ressep,
    )

    if subset:
        segid, start, stop = subset[0]
        logger.info("Using a subset of {} between {:d} and {:d}".format(
            segid, start, stop))
        kb: pd.DataFrame = kb.loc[segid].loc[start:stop]
        D_info["kb"] = kb.copy(deep=True)
        D_info["subset"] = subset[0]
        D_info["Npos"] = kb.index.size

    if kpos < 1:
        kpos = None
    fs: FluctSCA = FluctSCA(n_components=kpos, max_iter=ntrials,
                                  whiten=whiten, stddev=std, tol=tol,
                                  method=ictype)
    Vpica = fs.fit_transform(kb.T)
    logging.info(f"Using {fs.n_components_:d} sectors.")
    D_sca = dict(
        Csca=fs.Vfeatures_,
        Lrand=fs.random_ if kpos is None else None,
        ntrials=ntrials
    )

    ics, icsize, sortedpos, cutoff, scaled_pd, pdf = fluctsca.icList(
        Vpica, fs.n_components_, fs.Vfeatures_, p_cut=pcut)
    percentage: float = len(sortedpos) / D_info["npos"] * 100
    logger.info(f"{len(sortedpos):d} residues are within {_kpos:d} "
                f"sectors: {percentage:.2f}%")

    logger.info("Calculating the ICA for the windows.")
    D_sector = dict(
        kpos=fs.n_components_,
        Vsca=fs.Vfeatures_,
        Lsca=fs.eigenvalue_,
        Vrica=Vpica,
        Wrica=fs.Wfica_,
        ics=ics,
        icsize=icsize,
        sortedpos=sortedpos,
        cutoff=cutoff,
        scaled_pd=scaled_pd,
        pdf=pdf,
        Usca=fs.Usamples_,
        Urica=fs.Ufica_,
        Uwica=fs.Usica_,
        Wwica=fs.Wsica_,
    )

    D = dict(info=D_info, sca=D_sca, sector=D_sector)
    with open(click.format_filename(output), mode="wb") as dbf:
        logger.info("Saving data to {}".format(click.format_filename(output)))
        pickle.dump(D, dbf, protocol=pickle.HIGHEST_PROTOCOL)
