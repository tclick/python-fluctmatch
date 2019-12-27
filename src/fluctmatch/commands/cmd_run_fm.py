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
import importlib
import logging
import logging.config
from distutils.spawn import find_executable
from pathlib import Path
from typing import MutableMapping

import click

import fluctmatch.fluctmatch.plugins
from .. import iter_namespace


@click.command("run_fm", short_help="Run fluctuation matching.")
@click.option(
    "-s",
    "topology",
    metavar="FILE",
    default=Path.cwd() / "md.tpr",
    show_default=True,
    type=click.Path(exists=False, file_okay=True, resolve_path=True),
    help="Gromacs topology file (e.g., tpr gro g96 pdb brk ent)",
)
@click.option(
    "-f",
    "trajectory",
    metavar="FILE",
    default=Path.cwd() / "md.xtc",
    show_default=True,
    type=click.Path(exists=False, file_okay=True, resolve_path=True),
    help="Trajectory file (e.g. xtc trr dcd)",
)
@click.option(
    "-l",
    "--logfile",
    metavar="LOG",
    show_default=True,
    default=Path.cwd() / "charmmfm.log",
    type=click.Path(exists=False, file_okay=True, resolve_path=True),
    help="Log file",
)
@click.option(
    "-o",
    "outdir",
    metavar="DIR",
    default=Path.cwd(),
    show_default=True,
    type=click.Path(exists=False, file_okay=False, resolve_path=True),
    help="Directory",
)
@click.option(
    "-e",
    "--exec",
    "nma_exec",
    metavar="FILE",
    envvar="CHARMMEXEC",
    default=Path(find_executable("charmm")),
    show_default=True,
    type=click.Path(exists=False, file_okay=True, resolve_path=True),
    help="CHARMM executable file",
)
@click.option(
    "-t",
    "--temperature",
    metavar="TEMP",
    type=click.FLOAT,
    default=300.0,
    show_default=True,
    help="Temperature of simulation",
)
@click.option(
    "--max",
    "max_cycles",
    metavar="MAXCYCLES",
    type=click.IntRange(1, None, clamp=True),
    default=300,
    show_default=True,
    help="maximum number of fluctuation matching cycles",
)
@click.option(
    "--min",
    "min_cycles",
    metavar="MINCYCLES",
    type=click.IntRange(1, None, clamp=True),
    default=200,
    show_default=True,
    help="minimum number of fluctuation matching cycles",
)
@click.option(
    "--tol",
    metavar="TOL",
    type=click.FLOAT,
    default=1.0e-4,
    show_default=True,
    help="Tolerance level between simulations",
)
@click.option(
    "--force",
    "force_tol",
    metavar="FORCE",
    type=click.FLOAT,
    default=2.0e-1,
    show_default=True,
    help="force constants <= force tolerance become zero after min_cycles",
)
@click.option(
    "-p",
    "--prefix",
    metavar="PREFIX",
    default="fluctmatch",
    show_default=True,
    type=click.STRING,
    help="Prefix for filenames",
)
@click.option(
    "-c",
    "--charmm",
    "charmm_version",
    metavar="VERSION",
    default=41,
    show_default=True,
    type=click.IntRange(27, None, clamp=True),
    help="CHARMM version",
)
@click.option(
    "--extended / --standard",
    "extended",
    default=True,
    help="Output using the extended or standard columns",
)
@click.option(
    "--nb / --no-nb",
    "nonbonded",
    default=True,
    help="Include nonbonded section in CHARMM parameter file",
)
@click.option(
    "--resid / --no-resid",
    "resid",
    default=True,
    help="Include segment IDs in internal coordinate files",
)
@click.option(
    "--restart",
    is_flag=True,
    help="Restart simulation"
)
def cli(topology, trajectory, logfile, outdir, nma_exec, temperature,
        max_cycles, min_cycles, tol, force_tol, prefix, charmm_version,
        extended, resid, nonbonded, restart):
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

    FLUCTMATCH: MutableMapping = {
        name.split(".")[-1].upper(): importlib.import_module(name).Model
        for _, name, _
        in iter_namespace(fluctmatch.fluctmatch.plugins)
    }

    kwargs = dict(
        prefix=prefix,
        outdir=outdir,
        temperature=temperature,
        charmm_version=charmm_version,
        extended=extended,
        resid=resid,
        nonbonded=nonbonded,
    )
    cfm = FLUCTMATCH[nma_exec.name.upper()](topology, trajectory, **kwargs)

    logger.info("Initializing the parameters.")
    cfm.initialize(nma_exec=nma_exec, restart=restart)
    logger.info("Running fluctuation matching.")
    cfm.run(nma_exec=nma_exec, tol=tol, max_cycles=max_cycles,
            min_cycles=min_cycles, force_tol=force_tol)
    logger.info("Fluctuation matching successfully completed.")
