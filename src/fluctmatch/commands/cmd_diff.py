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

from fluctmatch.analysis import paramtable


@click.command("diff", short_help="Calculate differences between two tables.")
@click.option(
    "-l",
    "--logfile",
    metavar="LOG",
    show_default=True,
    default=Path.cwd() / "diff.log",
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
    "table1", metavar="TABLE1", type=click.Path(exists=True, resolve_path=True)
)
@click.argument(
    "table2", metavar="TABLE2", type=click.Path(exists=True, resolve_path=True)
)
def cli(logfile, outdir, ressep, table1, table2):
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

    logger.info("Loading %s", table1)
    table_1 = paramtable.ParamTable(ressep=ressep)
    table_1.from_file(table1)
    logger.info("%s loaded", table1)

    logger.info("Loading %s", table2)
    table_2 = paramtable.ParamTable(ressep=ressep)
    table_2.from_file(table2)
    logger.info("%s loaded", table2)

    d_table = table_1 - table_2
    d_perres = table_1.per_residue.subtract(table_2.per_residue, fill_value=0.0)
    d_interactions = table_1.interactions.subtract(table_2.interactions, fill_value=0.0)

    filename = Path(outdir) / "dcoupling.txt"
    with open(filename, mode="w") as output:
        logger.info("Writing table differences to %s", filename)
        d_table.to_csv(
            output, header=True, index=True, float_format="%.4f", encoding="utf-8"
        )
        logger.info("Table written successfully.")

    filename = Path(outdir) / "dperres.txt"
    with open(filename, mode="w") as output:
        logger.info("Writing per residue differences to %s", filename)
        d_perres.to_csv(
            output, header=True, index=True, float_format="%.4f", encoding="utf-8"
        )
        logger.info("Table written successfully.")

    filename = Path(outdir) / "dinteractions.txt"
    with open(filename, mode="w") as output:
        logger.info("Writing residue-residue differences to %s", filename)
        d_interactions.to_csv(
            output, header=True, index=True, float_format="%.4f", encoding="utf-8"
        )
        logger.info("Table written successfully.")
