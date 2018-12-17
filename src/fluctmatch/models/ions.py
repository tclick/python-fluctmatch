# -*- coding: utf-8 -*-
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
from collections import OrderedDict
from typing import Dict, Generator

from MDAnalysis.core import topologyattrs
from .base import ModelBase
from .selection import *


class SolventIons(ModelBase):
    """Include ions within the solvent.
    """
    model: str = "SOLVENTIONS"
    describe: str = "Common ions within solvent (Li K Na F Cl Br I)"
    _mapping: Dict = OrderedDict()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._mapping["ION"]: str = "name LI LIT K NA F CL BR I"

        kwargs["guess_bonds"]: bool = False
        kwargs["mapping"]: Dict[str, str] = self._mapping
        self._initialize(*args, **kwargs)
        resnames: np.ndarray = np.unique(self.residues.resnames)
        restypes: Dict[str, int] = {
            k: v
            for k, v in zip(resnames, np.arange(resnames.size) + 10)
        }
        self.atoms.types: np.ndarray = np.asarray(
            [restypes[atom.resname] for atom in self.atoms]
        )

    def _add_bonds(self):
        self._topology.add_TopologyAttr(topologyattrs.Bonds([]))
        self._generate_from_topology()


class BioIons(ModelBase):
    """Select ions normally found within biological systems.
    """
    model: str = "BIOIONS"
    describe: str = "Common ions found near proteins (Mg Ca Mn Fe Cu Zn Ag)"
    _mapping: Dict[str, str] = OrderedDict()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._mapping["ions"]: str = "bioion"

        kwargs["guess_bonds"]: bool = False
        kwargs["mapping"]: Dict[str, str] = self._mapping
        self._initialize(*args, **kwargs)
        resnames: np.ndarray = np.unique(self.residues.resnames)
        restypes: Dict[str, int] = {
            k: v
            for k, v in zip(resnames,
                            np.arange(resnames.size) + 20)
        }
        self.atoms.types: np.ndarray = np.asarray(
            [restypes[atom.resname] for atom in self.atoms]
        )

    def _add_bonds(self):
        self._topology.add_TopologyAttr(topologyattrs.Bonds([]))
        self._generate_from_topology()


class NobleAtoms(ModelBase):
    """Select atoms column VIII of the periodic table.
    """
    model: str = "NOBLE"
    describe: str = "Noble gases (He Ne Kr Xe)"
    _mapping: Dict[str, str] = OrderedDict()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._mapping["noble"]: str = "name HE NE KR XE"

        kwargs["guess_bonds"]: bool = False
        kwargs["mapping"]: Dict[str, str] = self._mapping
        self._initialize(*args, **kwargs)
        resnames: np.ndarray = np.unique(self.residues.resnames)
        restypes: Dict[str, int] = {
            k: v
            for k, v in zip(resnames,
                            np.arange(resnames.size) + 40)
        }
        self.atoms.types: np.ndarray = np.asarray(
            [restypes[atom.resname] for atom in self.atoms]
        )

    def _add_bonds(self):
        self._topology.add_TopologyAttr(topologyattrs.Bonds([]))
        self._generate_from_topology()
