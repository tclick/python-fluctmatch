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

from fluctmatch.analysis.paramstats import ParamStats
from fluctmatch.analysis.paramtable import ParamTable


@click.command("stats", short_help="Calculate statistics of a table.")
@click.option(
    "-l",
    "--logfile",
    metavar="LOG",
    show_default=True,
    default=Path.cwd() / "convert.log",
    type=click.Path(exists=False, file_okay=True, resolve_path=True),
    help="Log file",
)
@click.option("-s", "--stats", is_flag=True, help="Calculate statistics of tables")
@click.option("-g", "--hist", is_flag=True, help="Calculate histograms of tables")
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
@click.option(
    "-t",
    "--type",
    "tbltype",
    metavar="TYPE",
    default="Kb",
    show_default=True,
    type=click.Choice(["Kb", "b0"]),
    help="Force constant or equilibrium distance",
)
@click.argument(
    "table",
    metavar="TABLE",
    type=click.Path(exists=True, file_okay=True, resolve_path=True),
)
def cli(logfile, stats, hist, outdir, ressep, tbltype, table):
    # Setup logger
    logging.config.dictConfig(
        dict(
            version=1,
            disable_existing_loggers=False,  # this fixes the problem
            formatters=dict(
                standard={
                    "class": "logging.Formatter",
                    "format": "%(name)-12s %(levelname)-8s %(message)s",
                },
                detailed={
                    "class": "logging.Formatter",
                    "format": (
                        "%(asctime)s %(name)-15s %(levelname)-8s " "%(message)s"
                    ),
                    "datefmt": "%m-%d-%y %H:%M",
                },
            ),
            handlers=dict(
                console={
                    "class": "logging.StreamHandler",
                    "level": "INFO",
                    "formatter": "standard",
                },
                file={
                    "class": "logging.FileHandler",
                    "filename": logfile,
                    "level": "INFO",
                    "mode": "w",
                    "formatter": "detailed",
                },
            ),
            root=dict(level="INFO", handlers=["console", "file"]),
        )
    )
    logger: logging.Logger = logging.getLogger(__name__)

    logger.info(f"Reading {table}")
    pt = ParamTable(ressep=ressep)
    pt.from_file(table)
    ps = ParamStats(pt)
    outdir = Path(outdir)

    if stats:
        filename = outdir / "_".join((tbltype.lower(), "table", "stats.csv"))
        with open(filename, mode="w") as stat_file:
            logger.info(f"Writing table statistics to {filename}")
            ps.table_stats().to_csv(
                stat_file,
                header=True,
                index=True,
                float_format="%.4f",
                encoding="utf-8",
            )
            logger.info("Table successfully written.")

        if tbltype == "Kb":
            filename = outdir / "interaction_stats.csv"
            with open(filename, mode="w") as stat_file:
                logger.info(f"Writing residue-residue statistics to {filename}")
                ps._table._ressep = 0
                ps.interaction_stats().to_csv(
                    stat_file,
                    header=True,
                    index=True,
                    float_format="%.4f",
                    encoding="utf-8",
                )
                logger.info("Table successfully written.")

            filename = outdir / "residue_stats.csv"
            with open(filename, mode="w") as stat_file:
                logger.info(f"Writing individual residue statistics " f"to {filename}")
                ps._table._ressep = ressep
                ps.residue_stats().to_csv(
                    stat_file,
                    header=True,
                    index=True,
                    float_format="%.4f",
                    encoding="utf-8",
                )
                logger.info("Table successfully written.")

    if hist:
        filename = outdir / "_".join((tbltype.lower(), "table", "hist.csv"))
        with open(filename, mode="w") as stat_file:
            logger.info(f"Writing table histogram to {filename}")
            ps.table_hist().to_csv(
                stat_file, index=True, float_format="%.4f", encoding="utf-8"
            )
            logger.info("Table successfully written.")

        if tbltype == "Kb":
            filename = outdir / "interaction_hist.csv"
            with open(filename, mode="w") as stat_file:
                logger.info(f"Writing residue-residue histogram to {filename}")
                ps._table._ressep = 0
                ps.interaction_hist().to_csv(
                    stat_file, index=True, float_format="%.4f", encoding="utf-8"
                )
                logger.info("Table successfully written.")

            filename = outdir / "residue_hist.csv"
            with open(filename, mode="w") as stat_file:
                logger.info(f"Writing individual residue histogram to {filename}")
                ps._table._ressep = ressep
                ps.residue_hist().to_csv(
                    stat_file, index=True, float_format="%.4f", encoding="utf-8"
                )
                logger.info("Table successfully written.")
