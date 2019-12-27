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
from pathlib import Path

import click
from MDAnalysis.lib.util import filename

from fluctmatch.analysis import paramtable


@click.command("table",
               short_help="Create a table from individual parameter files.")
@click.option(
    "-d",
    "--datadir",
    "data_dir",
    metavar="DIR",
    default=Path.cwd() / "data",
    show_default=True,
    type=click.Path(exists=False, file_okay=False, resolve_path=True),
    help="Data directory",
)
@click.option(
    "-l",
    "--logfile",
    metavar="LOG",
    show_default=True,
    default=Path.cwd() / "table.log",
    type=click.Path(exists=False, file_okay=True, resolve_path=True),
    help="Log file",
)
@click.option(
    "-o",
    "--outdir",
    metavar="OUTDIR",
    default=Path.cwd(),
    show_default=True,
    type=click.Path(exists=False, file_okay=False, resolve_path=True),
    help="Directory",
)
@click.option(
    "-p",
    "--prefix",
    metavar="PREFIX",
    default="cg",
    show_default=True,
    type=click.STRING,
    help="Prefix for filenames",
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
@click.option(
    "-r",
    "--ressep",
    metavar="RESSEP",
    default=3,
    show_default=True,
    type=click.IntRange(0, None, clamp=True),
    help="Number of residues to exclude in I,I+r",
)
@click.option("-v", "--verbose", is_flag=True)
def cli(data_dir, logfile, outdir, prefix, tbltype, ressep, verbose):
    pt = paramtable.ParamTable(prefix=prefix, tbltype=tbltype, ressep=ressep,
                               datadir=data_dir)
    outdir = Path(outdir)

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

    pt.run(verbose=verbose)

    # Write the various tables to different files.
    fn = outdir / filename(tbltype.lower(), ext="csv", keep=True)
    pt.write(fn)

    if tbltype == "Kb":
        fn = outdir / filename("perres", ext="csv")
        with open(fn, mode="w") as output:
            logger.info("Writing per-residue data to {}.".format(fn))
            pt.per_residue.to_csv(output, header=True, index=True,
                                  float_format="%.4f", encoding="utf-8")
            logger.info("Table successfully written.")

        fn = outdir / filename("interactions", ext="csv")
        with open(fn, mode="w") as output:
            logger.info("Writing interactions to {}.".format(fn))
            pt.interactions.to_csv(output, header=True, index=True,
                                   float_format="%.4f", encoding="utf-8")
            logger.info("Table successfully written.")
