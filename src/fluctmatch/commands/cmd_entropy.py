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
from os import path
from pathlib import Path

import click

from fluctmatch.analysis import entropy


@click.command("entropy",
               short_help="Calculate the Shannon entropy of residues.")
@click.option(
    "-l",
    "--logfile",
    metavar="LOG",
    show_default=True,
    default=Path.cwd() / "entropy.log",
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
    help="Separation between residues (I,I+n)",
)
@click.argument(
    "table",
    metavar="TABLE",
    type=click.Path(exists=True, file_okay=True, resolve_path=True),
)
def cli(logfile, outdir, ressep, table):
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

    logger.info("Loading {}".format(table))
    ent_table = entropy.Entropy(table, ressep=ressep)

    filename = Path(outdir) / "coupling.entropy.txt"
    with open(filename, mode="w") as output:
        logger.info(f"Writing coupling entropy to {filename}")
        ent_table.coupling_entropy().to_csv(output, index=True, header=True,
                                            float_format="%.4f",
                                            encoding="utf-8")
        logger.info("Table written successfully.")

    filename = Path(outdir) / "relative.entropy.txt"
    with open(filename, mode="w") as output:
        logger.info(f"Writing relative entropy to {filename}")
        ent_table.relative_entropy().to_csv(output, index=True, header=True,
                                            float_format="%.4f",
                                            encoding="utf-8")
        logger.info("Table written successfully.")

    filename = Path(outdir) / "windiff.entropy.txt"
    with open(filename, mode="w") as output:
        logger.info(f"Writing entropy for window difference to {filename}")
        ent_table.windiff_entropy().to_csv(output, index=True, header=True,
                                           float_format="%.4f",
                                           encoding="utf-8")
        logger.info("Table written successfully.")
