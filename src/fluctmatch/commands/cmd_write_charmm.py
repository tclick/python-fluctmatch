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

import MDAnalysis as mda
import click

from fluctmatch.libs import fluctmatch as fmutils


@click.command("write_charmm", short_help="Write various CHARMM files.")
@click.option(
    "-s",
    "topology",
    metavar="FILE",
    type=click.Path(exists=True, file_okay=True, resolve_path=True),
    help="Gromacs topology file (e.g., tpr gro g96 pdb brk ent)",
)
@click.option(
    "-f",
    "trajectory",
    metavar="FILE",
    type=click.Path(exists=True, file_okay=True, resolve_path=True),
    help="Trajectory file (e.g. xtc trr dcd)",
)
@click.option(
    "-l",
    "--logfile",
    metavar="LOG",
    show_default=True,
    default=Path.cwd() / "write_charmm.log",
    type=click.Path(exists=False, file_okay=True, resolve_path=True),
    help="Log file",
)
@click.option(
    "-o",
    "outdir",
    metavar="DIR",
    type=click.Path(exists=True, file_okay=False, resolve_path=True),
    help="Trajectory file (e.g. xtc trr dcd)",
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
    "--no-nb",
    "nonbonded",
    is_flag=True,
    help="Include nonbonded section in CHARMM parameter file",
)
@click.option(
    "--no-cmap", "cmap", is_flag=True,
    help="Include CMAP section in CHARMM PSF file"
)
@click.option(
    "--no-cheq",
    "cheq",
    is_flag=True,
    help="Include charge equilibrium section in CHARMM PSF file",
)
@click.option("--write", "write_traj", is_flag=True,
              help="Convert the trajectory file")
def cli(topology, trajectory, logfile, outdir, prefix, charmm_version, extended,
        cmap, cheq, nonbonded, write_traj):
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

    kwargs = dict(
        outdir=outdir,
        prefix=prefix,
        charmm_version=charmm_version,
        extended=extended,
        cmap=not cmap,
        cheq=not cheq,
        nonbonded=not nonbonded,
        write_traj=write_traj,
    )
    universe = mda.Universe(topology, trajectory)
    logger.info("Writing CHARMM files.")
    fmutils.write_charmm_files(universe, **kwargs)
