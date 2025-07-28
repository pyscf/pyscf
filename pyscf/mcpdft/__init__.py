#!/usr/bin/env python
# Copyright 2014-2022 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Lahh dee dah
'''
Multi-configuration pair-density functional theory
==================================================

Simple usage::

    >>> from pyscf import gto, scf, mcpdft
    >>> mol = gto.M(atom='N 0 0 0; N 0 0 1', basis='def2-tzvp')
    >>> mf = scf.RHF(mol).run ()
    >>> mc = mcpdft.CASSCF (mf, 'tPBE', 6, 6)
    >>> mc.run()
'''


import copy
from pyscf.mcpdft.mcpdft import get_mcpdft_child_class
from pyscf.mcpdft.otfnal import make_hybrid_fnal as hyb
from pyscf import mcscf, gto
from pyscf.lib import logger
from pyscf.mcscf import mc1step, casci

def _sanity_check_of_mol(mc_or_mf_or_mol):
    '''
    Sanity check to ensure input is a mol object, not a cell object.
    Raises an error for cell objects.
    '''
    from pyscf.pbc import gto as pbcgto
    if isinstance(mc_or_mf_or_mol, (mc1step.CASSCF, casci.CASCI)):
        mol = mc_or_mf_or_mol._scf.mol
    elif isinstance(mc_or_mf_or_mol, gto.Mole):
        mol = mc_or_mf_or_mol
    else:
        mol = mc_or_mf_or_mol.mol

    if isinstance(mol, pbcgto.cell.Cell):
        raise NotImplementedError("MCPDFT not implemented for PBC")

# NOTE: As of 02/06/2022, initializing PySCF mcscf classes with a symmetry-enabled molecule
# doesn't work.

def _MCPDFT (mc_class, mc_or_mf_or_mol, ot, ncas, nelecas, ncore=None, frozen=None,
             **kwargs):
    # Raise an error if mol is a cell object.
    _sanity_check_of_mol(mc_or_mf_or_mol)
    if isinstance (mc_or_mf_or_mol, (mc1step.CASSCF, casci.CASCI)):
        mc0 = mc_or_mf_or_mol
        mf_or_mol = mc_or_mf_or_mol._scf
    else:
        mc0 = None
        mf_or_mol = mc_or_mf_or_mol
    if isinstance (mf_or_mol, gto.Mole) and mf_or_mol.symmetry:
        logger.warn (mf_or_mol,
                     'Initializing MC-SCF with a symmetry-adapted Mole object may not work!')
    if frozen is not None: mc1 = mc_class (mf_or_mol, ncas, nelecas, ncore=ncore, frozen=frozen)
    else: mc1 = mc_class (mf_or_mol, ncas, nelecas, ncore=ncore)
    mc2 = get_mcpdft_child_class (mc1, ot, **kwargs)
    if mc0 is not None:
        mc2.verbose = mc0.verbose
        mc2.stdout = mc0.stdout
        mc2.mo_coeff = mc_or_mf_or_mol.mo_coeff.copy ()    
        mc2.ci = copy.deepcopy (mc_or_mf_or_mol.ci)
        mc2.converged = mc0.converged
    return mc2

def CASSCFPDFT (mc_or_mf_or_mol, ot, ncas, nelecas, ncore=None, frozen=None,
                **kwargs):
    return _MCPDFT (mcscf.CASSCF, mc_or_mf_or_mol, ot, ncas, nelecas, ncore=ncore, frozen=frozen,
                    **kwargs)

def CASCIPDFT (mc_or_mf_or_mol, ot, ncas, nelecas, ncore=None, **kwargs):
    return _MCPDFT (mcscf.CASCI, mc_or_mf_or_mol, ot, ncas, nelecas, ncore=ncore,
                    **kwargs)

CASSCF=CASSCFPDFT
CASCI=CASCIPDFT

class MultiStateMCPDFTSolver :
    pass
    # tag


# Monkeypatch for double prop folders
# TODO: more elegant solution
import os, warnings
# Search for PySCF-Forge via mcdcft
try:
    from pyscf import mcdcft
    mypath = os.path.dirname (os.path.dirname (os.path.abspath (mcdcft.__file__)))
    myproppath = os.path.join (mypath, 'prop')
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="Module.*is under testing")
        from pyscf import prop
    prop.__path__.append(myproppath)
    prop.__path__ = list(set(prop.__path__))
except (ModuleNotFoundError,ImportError):
    pass
