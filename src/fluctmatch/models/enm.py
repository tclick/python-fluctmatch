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
import numpy as np
from MDAnalysis.core import topologyattrs
from MDAnalysis.lib.distances import distance_array, self_capped_distance
from fluctmatch.fluctmatch import utils as fmutils
from fluctmatch.models.base import (
    ModelBase,
    rename_universe,
)


class Enm(ModelBase):
    """Convert a basic coarse-grain universe into an elastic-network model.

    Determines the interactions between beads via distance cutoffs `rmin` and
    `rmax`. The atoms and residues are also renamed to prevent name collision
    when working with fluctuation matching.
    """
    model = "ENM"
    describe = "Elastic network model"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._rmin: float = kwargs.get("rmin", 0.)
        self._rmax: float = kwargs.get("rmax", 10.)
        self._initialize(*args, **kwargs)

    def __repr__(self) -> str:
        message: str = f"<CG Universe with {self.atoms.n_atoms} beads"
        try:
            message += f" and {len(self._topology.bonds.values)} bonds"
        except AttributeError as exc:
            pass
        finally:
            message += ">"
        return message

    def _initialize(self, *args, **kwargs):
        self.__dict__.update(self.atu.__dict__)

        rename_universe(self)
        charges: bool = kwargs.get("charges", False)
        if not charges:
            self._topology.add_TopologyAttr(
                topologyattrs.Charges(np.zeros(self.atoms.n_atoms))
            )
        self._topology.add_TopologyAttr(
            topologyattrs.Atomtypes(np.arange(self.atoms.n_atoms) + 1)
        )
        self._topology.add_TopologyAttr(topologyattrs.Angles([]))
        self._topology.add_TopologyAttr(topologyattrs.Dihedrals([]))
        self._topology.add_TopologyAttr(topologyattrs.Impropers([]))
        self._generate_from_topology()

        self._add_bonds()
        if kwargs.get("guess_angles", False):
            self._add_angles()
            self._add_dihedrals()
            self._add_impropers()

    def _add_bonds(self):
        positions: np.ndarray = fmutils.AverageStructure(self.atu.atoms).run().result
        pairs, _ = self_capped_distance(
            positions, self._rmax, min_cutoff=self._rmin
        )
        bonds = topologyattrs.Bonds(np.unique(pairs, axis=0))
        self._topology.add_TopologyAttr(bonds)
        self._generate_from_topology()

    @property
    def rmin(self) -> float:
        """The minimum distance required to define a bond interaction.

        Returns
        -------
        float
        """
        return self._rmin

    @rmin.setter
    def rmin(self, distance: float):
        """Set the minimum distance required for a bond definition.

        Parameters
        ----------
        distance : float
            Minimum distance between beads
        """
        self._rmin: float = distance

    @property
    def rmax(self) -> float:
        """The maximum distance required to define a bond interaction.

        Returns
        -------
        float
        """
        return self._rmax

    @rmax.setter
    def rmax(self, distance: float):
        """Set the maximum distance required for a bond definition.

        Parameters
        ----------
        distance : float
            Maximum distance between beads
        """
        self._rmax: float = distance
