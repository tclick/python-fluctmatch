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
import os
import time

from MDAnalysis.coordinates.base import (_Readermeta, _Writermeta, IOBase)

from ..lib.register import register_reader, register_writer


class TopologyReaderBase(IOBase, metaclass=_Readermeta):
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        register_reader(cls)

    def read(self):  # pragma: no cover
        """Read the file"""
        raise NotImplementedError("Override this in each subclass")


class TopologyWriterBase(IOBase, metaclass=_Writermeta):
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        register_writer(cls)

    def __init__(self):
        self.title: str = (
            f"""
            * Created by fluctmatch on {time.asctime(time.localtime())}
            * User: {os.environ["USER"]}
            """
        )

    def write(self, selection):  # pragma: no cover
        # type: (object) -> object
        """Write selection at current trajectory frame to file.

        Parameters
        ----------
        selection : AtomGroup
             group of atoms to be written

        """
        raise NotImplementedError("Override this in each subclass")
