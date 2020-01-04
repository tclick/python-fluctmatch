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
from pathlib import Path

import click
import numpy as np

from fluctmatch.analysis import paramtable

logger: logging.Logger = logging.getLogger(__name__)


@click.command(
    "normdiff", short_help="Normalize the differences between t and t - dt/2."
)
@click.option(
    "-l",
    "--logfile",
    metavar="LOG",
    show_default=True,
    default=Path.cwd() / "normdiff.log",
    type=click.Path(exists=False, file_okay=True, resolve_path=True),
    help="Log file",
)
@click.option(
    "-o",
    "--outdir",
    metavar="OUTDIR",
    default=Path.cwd(),
    show_default=True,
    type=click.Path(exists=True, file_okay=False, resolve_path=True),
    help="Directory",
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
@click.argument(
    "kb",
    metavar="kb_table",
    type=click.Path(exists=True, file_okay=True, resolve_path=True),
)
@click.argument(
    "b0",
    metavar="b0_table",
    type=click.Path(exists=True, file_okay=True, resolve_path=True),
)
def cli(logfile, outdir, ressep, kb, b0):
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

    resgrp = ["segidI", "resI"]

    logger.info("Loading coupling strength table.")
    kb_table = paramtable.ParamTable(ressep=ressep)
    kb_table.from_file(kb)
    kb_table = kb_table.table.copy(deep=True)
    logger.info("Table loaded successfully.")

    logger.info("Loading distance table.")
    b0_table = paramtable.ParamTable(ressep=ressep)
    b0_table.from_file(b0)
    b0_table = b0_table.table.copy(deep=True)
    logger.info("Table loaded successfully.")

    idx = kb_table == 0.0
    maxkb = np.maximum(
        kb_table[kb_table.columns[1:].values],
        kb_table[kb_table.columns[:-1]].values
    )

    maxkb[maxkb == 0.0] = np.NaN
    kb_table = kb_table.diff(axis=1).dropna(axis=1) / maxkb
    kb_table.columns = kb_table.columns.astype(np.int32) - 1

    count = kb_table[kb_table.abs() > 0.0].groupby(level=resgrp).count()
    kb_table = kb_table.groupby(level=resgrp).sum() / count
    kb_table.fillna(0.0, inplace=True)

    # Calculate the r.m.s.d. of equilibrium distances between sites.
    b0_table = b0_table.diff(axis=1).dropna(axis=1) / maxkb
    b0_table[idx] = 0.0
    b0_table = 0.5 * b0_table.pow(2).groupby(level=resgrp).sum()
    b0_table = b0_table.apply(np.sqrt)

    filename = Path(outdir) / "normed_kb.txt"
    with open(filename, mode="w") as output:
        logger.info(f"Writing normed coupling strengths to {filename}")
        kb_table.to_csv(output, header=True, index=True, float_format="%.4f",
                        encoding="utf-8")
        logger.info("Table written successfully.")

    filename = Path(outdir) / "normed_b0.txt"
    with open(filename, mode="w") as output:
        logger.info(f"Writing normed distances to {filename}")
        b0_table.to_csv(output, header=True, index=True, float_format="%.4f",
                        encoding="utf-8")
        logger.info("Table written successfully.")
