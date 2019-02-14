# -*- coding: utf-8 -*-
#
#  python-fluctmatch -
#  Copyright (c) 2019 Timothy H. Click, Ph.D.
#
#  All rights reserved.
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:
#
#  Redistributions of source code must retain the above copyright notice, this
#  list of conditions and the following disclaimer.
#
#  Redistributions in binary form must reproduce the above copyright notice,
#  this list of conditions and the following disclaimer in the documentation
#  and/or other materials provided with the distribution.
#
#  Neither the name of the author nor the names of its contributors may be used
#  to endorse or promote products derived from this software without specific
#  prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS “AS IS”
#  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
#  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
#  ARE DISCLAIMED. IN NO EVENT SHALL THE REGENTS OR CONTRIBUTORS BE LIABLE FOR
#  ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
#  DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
#  SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
#  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
#  OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
#  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
#  Timothy H. Click, Nixon Raj, and Jhih-Wei Chu.
#  Simulation. Meth Enzymology. 578 (2016), 327-342,
#  Calculation of Enzyme Fluctuograms from All-Atom Molecular Dynamics
#  doi:10.1016/bs.mie.2016.05.024.

from typing import ClassVar

import numpy as np
from MDAnalysis.core import selection, AtomGroup


class BioIonSelection(selection.Selection):
    """Contains atoms commonly found in proteins."""
    token: ClassVar[str] = "bioion"
    ion_atoms: np.ndarray = np.array(["MG", "CAL", "MN", "FE", 
                                      "CU", "ZN", "AG"])

    def __init__(self, parser, tokens):
        pass

    def apply(self, group: AtomGroup):
        mask: np.ndarray = np.in1d(group.names, self.ion_atoms)
        return group[mask].unique


class WaterSelection(selection.Selection):
    """Contains atoms commonly found in water."""
    token: ClassVar[str] = "water"
    water_atoms: np.ndarray = np.array(["OW", "HW1", "HW2", "MW"])

    def __init__(self, parser, tokens):
        pass

    def apply(self, group: AtomGroup):
        mask: np.ndarray = np.in1d(group.names, self.water_atoms)
        return group[mask].unique


class BackboneSelection(selection.BackboneSelection):
    """Contains all heavy atoms within a protein backbone including C-termini.
    """
    token: ClassVar[str] = "backbone"
    oxy_atoms: np.ndarray = np.array(["OXT", "OT1", "OT2"])
    bb_atoms = np.concatenate([selection.BackboneSelection.bb_atoms, oxy_atoms])

    def apply(self, group: AtomGroup):
        mask: np.ndarray = np.in1d(group.names, self.bb_atoms)
        mask &= np.in1d(group.resnames, self.prot_res)
        return group[mask].unique


class HBackboneSelection(BackboneSelection):
    """Includes all atoms found within a protein backbone including hydrogens.
    """
    token: ClassVar[str] = "hbackbone"
    hbb_atoms: np.ndarray = np.array([
        "H", "HN", "H1", "H2", "H3", "HT1", "HT2", 
        "HT3", "HA", "HA1", "HA2", "1HA", "2HA"
    ])
    bb_atoms = np.concatenate([BackboneSelection.bb_atoms, hbb_atoms])

    def apply(self, group: AtomGroup):
        mask: np.ndarray = np.in1d(group.names, self.bb_atoms)
        mask &= np.in1d(group.resnames, self.prot_res)
        return group[mask].unique


class CalphaSelection(selection.ProteinSelection):
    """Contains only the alpha-carbon of a protein."""
    token: ClassVar[str] = "calpha"
    calpha = np.array(["CA"])

    def apply(self, group: AtomGroup):
        mask: np.ndarray = np.in1d(group.names, self.calpha)
        mask &= np.in1d(group.resnames, self.prot_res)
        return group[mask].unique


class HCalphaSelection(CalphaSelection):
    """Contains the alpha-carbon and alpha-hydrogens of a protein."""
    token: ClassVar[str] = "hcalpha"
    hcalpha = np.array(["HA", "HA1", "HA2", "1HA", "2HA"])
    calpha = np.concatenate([CalphaSelection.calpha, hcalpha])

    def apply(self, group: AtomGroup):
        mask: np.ndarray = np.in1d(group.names, self.calpha)
        mask &= np.in1d(group.resnames, self.prot_res)
        return group[mask].unique


class CbetaSelection(selection.ProteinSelection):
    """Contains only the beta-carbon of a protein."""
    token: ClassVar[str] = "cbeta"
    cbeta = np.array(["CB"])

    def apply(self, group: AtomGroup):
        mask: np.ndarray = np.in1d(group.names, self.cbeta)
        mask &= np.in1d(group.resnames, self.prot_res)
        return group[mask].unique


class AmineSelection(selection.ProteinSelection):
    """Contains atoms within the amine group of a protein."""
    token: ClassVar[str] = "amine"
    amine = np.array(["N", "HN", "H", "H1", "H2", "H3", "HT1", "HT2", "HT3"])

    def apply(self, group: AtomGroup):
        mask: np.ndarray = np.in1d(group.names, self.amine)
        mask &= np.in1d(group.resnames, self.prot_res)
        return group[mask].unique


class CarboxylSelection(selection.ProteinSelection):
    """Contains atoms within the carboxyl group of a protein."""
    token: ClassVar[str] = "carboxyl"
    carboxyl = np.array(["C", "O", "OXT", "OT1", "OT2"])

    def apply(self, group: AtomGroup):
        mask: np.ndarray = np.in1d(group.names, self.carboxyl)
        mask &= np.in1d(group.resnames, self.prot_res)
        return group[mask].unique


class HSidechainSelection(HBackboneSelection):
    """Includes hydrogens on the protein sidechain."""
    token: ClassVar[str] = "hsidechain"

    def apply(self, group: AtomGroup):
        mask: np.ndarray = np.in1d(group.names, HBackboneSelection.bb_atoms,
                                   invert=True)
        mask &= np.in1d(group.resnames, self.prot_res)
        return group[mask].unique


class AdditionalNucleicSelection(selection.NucleicSelection):
    """Contains additional nucleic acid residues."""
    token: ClassVar[str] = "nucleic"

    def __init__(self, parser, tokens):
        super().__init__(parser, tokens)
        self.nucl_res = np.concatenate((self.nucl_res, ["OXG", "HPX", "DC35"]))

    def apply(self, group: AtomGroup):
        mask: np.ndarray = np.in1d(group.resnames, self.nucl_res)
        return group[mask].unique


class HNucleicSugarSelection(AdditionalNucleicSelection,
                             selection.NucleicSugarSelection):
    """Contains the additional atoms definitions for the sugar."""
    token: ClassVar[str] = "hnucleicsugar"

    def __init__(self, parser, tokens):
        super().__init__(parser, tokens)
        self.sug_atoms: np.ndarray = np.concatenate(
            (self.sug_atoms,
             np.array(["H1'", "O1'", "O2'", "H2'", "H2''",
                       "O3'", "H3'", "H3T", "H4'"])))

    def apply(self, group: AtomGroup):
        mask: np.ndarray = np.in1d(group.names, self.sug_atoms)
        mask &= np.in1d(group.resnames, self.nucl_res)
        return group[mask].unique


class HBaseSelection(AdditionalNucleicSelection, selection.BaseSelection):
    """Contains additional atoms on the base region of the nucleic acids."""
    token: ClassVar[str] = "hnucleicbase"

    def __init__(self, parser, tokens):
        super().__init__(parser, tokens)
        self.base_atoms: np.ndarray = np.concatenate(
            (self.base_atoms, ["O8", "H8", "H21", "H22", "H2", "O6", "H6",
                               "H61", "H62", "H41", "H42", "H5", "H51", "H52",
                               "H53", "H3", "H7"]))

    def apply(self, group: AtomGroup):
        mask: np.ndarray = np.in1d(group.names, self.base_atoms)
        mask &= np.in1d(group.resnames, self.nucl_res)
        return group[mask].unique


class NucleicPhosphateSelection(AdditionalNucleicSelection):
    """Contains the nucleic phosphate group including the C5'."""
    token: ClassVar[str] = "nucleicphosphate"
    phos_atoms: np.ndarray = np.array(
        ["P", "O1P", "O2P", "O5'", "C5'", "H5'", "H5''", "O5T", "H5T"])

    def apply(self, group: AtomGroup):
        mask: np.ndarray = np.in1d(group.names, self.phos_atoms)
        mask &= np.in1d(group.resnames, self.nucl_res)
        return group[mask].unique


class NucleicC2Selection(AdditionalNucleicSelection):
    """Contains the definition for the C3' region."""
    token: ClassVar[str] = "sugarC2"
    c3_atoms: np.ndarray = np.array(["C1'", "H1'", "C2'", "O2'", "H2'", "H2''"])

    def apply(self, group: AtomGroup):
        mask: np.ndarray = np.in1d(group.names, self.c3_atoms)
        mask &= np.in1d(group.resnames, self.nucl_res)
        return group[mask].unique


class NucleicC4Selection(AdditionalNucleicSelection):
    """Contains the definition for the C4' region."""
    token: ClassVar[str] = "sugarC4"
    c3_atoms: np.ndarray = np.array(["C3'", "O3'", "H3'", "H3T",
                                     "C4'", "O4'", "H4'"])

    def apply(self, group: AtomGroup):
        mask: np.ndarray = np.in1d(group.names, self.c3_atoms)
        mask &= np.in1d(group.resnames, self.nucl_res)
        return group[mask].unique


class BaseCenterSelection(AdditionalNucleicSelection):
    """Contains the central atoms (C4 and C5) on the base of the nuleic acid."""
    token: ClassVar[str] = "nucleiccenter"
    center_atoms: np.ndarray = np.array(["C4", "C5"])

    def apply(self, group: AtomGroup):
        mask: np.ndarray = np.in1d(group.names, self.center_atoms)
        mask &= np.in1d(group.resnames, self.nucl_res)
        return group[mask].unique
