#!/usr/bin/env python

from pyscf.mcscf import mc1step
from pyscf.mcscf import mc1step_symm
from pyscf.mcscf import casci
from pyscf.mcscf import casci_symm
from pyscf.mcscf import addons
from pyscf.mcscf import casci_uhf
from pyscf.mcscf import mc1step_uhf
from pyscf.mcscf.addons import *

def CASSCF(mol, mf, *args, **kwargs):
    if mol.symmetry:
        mc = mc1step_symm.CASSCF(mol, mf, *args, **kwargs)
    else:
        if 'RHF' in str(mf.__class__) or 'ROHF' in str(mf.__class__):
            mc = mc1step.CASSCF(mol, mf, *args, **kwargs)
        else:
            mc = mc1step_uhf.CASSCF(mol, mf, *args, **kwargs)
    return mc

def CASCI(mol, mf, *args, **kwargs):
    if mol.symmetry:
        mc = casci_symm.CASCI(mol, mf, *args, **kwargs)
    else:
        if 'RHF' in str(mf.__class__) or 'ROHF' in str(mf.__class__):
            mc = casci.CASCI(mol, mf, *args, **kwargs)
        else:
            mc = casci_uhf.CASCI(mol, mf, *args, **kwargs)
    return mc
