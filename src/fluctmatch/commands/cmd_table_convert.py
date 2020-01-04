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
import MDAnalysis as mda
import pandas as pd


@click.command(
    "table_convert",
    short_help="Transform an ENM IC table name to corresponding atoms.",
)
@click.option(
    "-l",
    "--logfile",
    metavar="LOG",
    show_default=True,
    default=Path.cwd() / "table_convert.log",
    type=click.Path(resolve_path=True),
    help="Log file",
)
@click.option(
    "-s1",
    "top1",
    metavar="FILE",
    default=Path.cwd() / "cg.xplor.psf",
    show_default=True,
    type=click.Path(exists=True, resolve_path=True),
    help="Topology file",
)
@click.option(
    "-s2",
    "top2",
    metavar="FILE",
    default=Path.cwd() / "fluctmatch.xplor.psf",
    show_default=True,
    type=click.Path(exists=True, resolve_path=True),
    help="Topology file",
)
@click.option(
    "-t",
    "--table",
    metavar="FILE",
    default=Path.cwd() / "kb.txt",
    show_default=True,
    type=click.Path(exists=True, resolve_path=True),
    help="Coordinate file",
)
@click.option(
    "-o",
    "--outfile",
    metavar="OUTFILE",
    default=Path.cwd() / "kb_aa.txt",
    show_default=True,
    type=click.Path(resolve_path=True),
    help="Table file",
)
def cli(logfile, top1, top2, table, outfile):
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

    cg = mda.Universe(top1)
    fluctmatch = mda.Universe(top2)
    convert = dict(zip(fluctmatch.atoms.names, cg.atoms.names))
    resnames = pd.DataFrame.from_records(
        zip(cg.residues.segids, cg.residues.resnums, cg.residues.resnames),
        columns=["segid", "res", "resn"],
    ).set_index(["segid", "res"])

    with open(table) as tbl:
        logger.info(f"Loading {table}.")
        constants = pd.read_csv(
            tbl, header=0, skipinitialspace=True, delim_whitespace=True
        )
        logger.info("Table loaded successfully.")

    # Transform assigned bead names to an all-atom designation.
    constants["I"] = constants["I"].apply(lambda x: convert[x])
    constants["J"] = constants["J"].apply(lambda x: convert[x])

    # Create lists of corresponding residues
    columns = ["segidI", "resI", "segidJ", "resJ"]
    resnI = []
    resnJ = []
    for segidI, resI, segidJ, resJ in constants[columns].values:
        resnI.append(resnames.loc[(segidI, resI)])
        resnJ.append(resnames.loc[(segidJ, resJ)])
    constants["resnI"] = pd.concat(resnI).values
    constants["resnJ"] = pd.concat(resnJ).values

    # Concatenate the columns
    cols = ["segidI", "resI", "resnI", "I", "segidJ", "resJ", "resnJ", "J"]
    constants.set_index(cols, inplace=True)

    with open(outfile, "w") as output:
        logger.info(f"Writing updated table to {outfile}.")
        constants = constants.to_csv(
            header=True, index=True, sep=" ", float_format="%.4f"
        )
        output.write(constants.encode())
        logger.info("Table written successfully.")
