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
from __future__ import (
    absolute_import,
    division,
    print_function,
    unicode_literals,
)

from os import path

from future.utils import native_str, PY3
from future.builtins import open

import click
import numpy as np
from scipy import linalg
from six.moves import cPickle
from sklearn.utils import extmath

from fluctmatch.analysis.paramtable import ParamTable
from fluctmatch.analysis import (
    fluctsca,
    scaTools,
)


@click.command(
    "sca",
    short_help="Statistical coupling analysis (SCA) on coupling strength"
)
@click.option(
    "-n",
    "--ntrials",
    default=100,
    type=click.IntRange(0, None, clamp=True),
    help="Number of random iterations (default: 100)"
)
@click.option(
    "--std",
    default=2,
    type=np.int,
    help="Number of std. deviations for beyond second eigenmode (default: 2)"
)
@click.option(
    "-k",
    "--kpos",
    default=0,
    type=click.IntRange(0, None, clamp=True),
    help="Number of eigenmodes (default: auto)"
)
@click.option(
    "-p",
    "--pcut",
    default=0.95,
    type=np.float,
    help="Cutoff value for sector selection"
)
@click.option(
    "-r",
    "--ressep",
    metavar="RESSEP",
    default=2,
    type=click.IntRange(0, None, clamp=True),
    help="Number of residues to exclude in I,I+r (default: 2)"
)
@click.option(
    "-o",
    "--output",
    default=path.join(path.curdir, "scafluct.db"),
    type=click.Path(
        exists=False,
        file_okay=False,
        resolve_path=True,
    ),
    help="Output filename (default: ./scafluct.db)"
)
@click.argument(
    "filename",
    type=click.Path(
        exists=True,
        file_okay=False,
        resolve_path=True,
    )
)
def calculate_scafluct(
    ntrials, std, kpos, pcut, ressep, output, filename
):
    # Load the table, separate by I,I+r, and if requested, create a subset.
    table = ParamTable(ressep=ressep)
    table.from_file(click.format_filename(filename))
    kb = table.interactions

    D_info = dict(
        kb=kb,
        ressep=ressep,
        subset=subset
    )

    # Calculate eigenvalues and eigenvectors for the time series with sign correction.
    U, Lsca, Vt = linalg.svd(kb, full_matrices=False)
    U, Vt = extmath.svd_flip(U, Vt, u_based_decision=True)

    # Sign correction similar to that in SCA
    Lrand = fluctsca.randomize(kb, ntrials=ntrials)
    Ucorrel = kb.values.dot(kb.T.values)
    Vcorrel = kb.values.T.dot(kb.values)
    D_sca = dict(
        U=U,
        Lsca=Lsca,
        Vt=Vt,
        Lrand=Lrand,
        Ucorrel=Ucorrel,
        Vcorrel=Vcorrel,
        ntrials=ntrials
    )

    # Determine the number of eigenmodes if kpos = 0
    _kpos = fluctsca.chooseKpos(Lsca, Lrand, stddev=std) if kpos == 0 else kpos
    click.echo("Selecting {:d} eigenmodes".format(_kpos))
    Ucorr = fluctsca.correlate(U, Lsca, kmax=_kpos)
    Vcorr = fluctsca.correlate(Vt.T, Lsca, kmax=_kpos)

    # Calculate IC sectors
    Uica, Wres = scaTools.rotICA(U, kmax=_kpos)
    Uics, Uicsize, Usortedpos, Ucutoff, Uscaled_pd, Upd = scaTools.icList(
        Uica, _kpos, Ucorrel, p_cut=pcut
    )
    Vica, Wtime = scaTools.rotICA(Vt.T, kmax=_kpos)
    Utica = Wtime.dot(U[:,:_kpos].T).T
    Vrica = Wres.dot(Vt[:_kpos]).T
    Vics, Vicsize, Vsortedpos, Vcutoff, Vscaled_pd, Vpd = scaTools.icList(
        Vica, _kpos, Vcorrel, p_cut=pcut
    )
    D_sector = dict(
        std=std,
        kpos=_kpos,
        Ucorr=Ucorr,
        Vcorr=Vcorr,
        Uica=Uica,
        Wres=Wres,
        Vica=Vica,
        Wtime=Wtime,
        Uics=Uics,
        Uicsize=Uicsize,
        Usortedpos=Usortedpos,
        Ucutoff=Ucutoff,
        Uscaled_pd=Uscaled_pd,
        Upd=Upd,
        Vics=Vics,
        Vicsize=Vicsize,
        Vsortedpos=Vsortedpos,
        Vcutoff=Vcutoff,
        Vscaled_pd=Vscaled_pd,
        Vpd=Vpd,
        Utica=Utica,
        Vrica=Vrica
    )

    D = dict(
        info=D_info,
        sca=D_sca,
        sector=D_sector
    )
    with open(click.format_filename(output), mode="wb") as dbf:
        click.echo("Saving data to {}".format(click.format_filename(output)))
        if PY3:
            click.echo(
                "Note: The saved file will be incompatible with Python 2."
            )
        cPickle.dump(D, dbf, protocol=cPickle.HIGHEST_PROTOCOL)