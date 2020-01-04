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
from MDAnalysis.lib.util import which

from fluctmatch.analysis import thermodynamics


@click.command("thermo", short_help="Calculate thermodynamic properties.")
@click.option(
    "-s",
    "topology",
    metavar="FILE",
    default="fluctmatch.xplor.psf",
    show_default=True,
    type=click.Path(exists=False, file_okay=True, resolve_path=False),
    help="Topology file (e.g., tpr gro g96 pdb brk ent psf)",
)
@click.option(
    "-f",
    "trajectory",
    metavar="FILE",
    default="cg.dcd",
    show_default=True,
    type=click.Path(exists=False, file_okay=True, resolve_path=False),
    help="Trajectory file (e.g. xtc trr dcd)",
)
@click.option(
    "-d",
    "datadir",
    metavar="DIR",
    default=Path.cwd() / "data",
    show_default=True,
    type=click.Path(exists=True, file_okay=False, resolve_path=True),
    help="Directory",
)
@click.option(
    "-l",
    "--logfile",
    metavar="LOG",
    show_default=True,
    default=Path.cwd() / "thermo.log",
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
    default=which("charmm"),
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
    "-c",
    "--charmm",
    "charmm_version",
    metavar="VERSION",
    default=41,
    show_default=True,
    type=click.IntRange(27, None, clamp=True),
    help="CHARMM version",
)
def cli(
    datadir,
    logfile,
    outdir,
    topology,
    trajectory,
    nma_exec,
    temperature,
    charmm_version,
):
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

    # Attempt to create the necessary subdirectory
    try:
        Path.mkdir(outdir, parents=True)
    except OSError:
        pass

    logger.info("Calculating thermodynamic properties.")
    logger.warning(
        "Depending upon the size of the system, this may take " "a while."
    )

    kwargs = dict(
        topology=topology,
        trajectory=trajectory,
        temperature=temperature,
        nma_exec=nma_exec,
        charmm_version=charmm_version,
    )
    thermodynamics.create_thermo_tables(datadir, outdir, **kwargs)
