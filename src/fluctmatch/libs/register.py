# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding: utf-8 -*-
# vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4
#
# fluctmatch --- https://github.com/tclick/python-fluctmatch
# Copyright (c) 2015-2017 The pySCA Development Team and contributors
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

from MDAnalysis import _MULTIFRAME_WRITERS, _READERS, _SINGLEFRAME_WRITERS

from .. import _DESCRIBE, _MODELS

logger: logging.Logger = logging.getLogger(__name__)


def register_model(target_class: object):
    _MODELS[target_class.model.upper()]: object = target_class


def register_description(target_class: object):
    _DESCRIBE[target_class.model.upper()]: str = target_class.describe


def register_reader(target_class: object):
    _READERS[target_class.format.upper()]: object = target_class

def register_writer(target_class: object):
    fmt: str = target_class.format.upper()

    # does the Writer support single and multiframe writing?
    single: bool = target_class.__dict__.get('singleframe', True)
    multi: bool = target_class.__dict__.get('multiframe', False)

    if single:
        _SINGLEFRAME_WRITERS[fmt]: object = target_class
    if multi:
        _MULTIFRAME_WRITERS[fmt]: object = target_class
