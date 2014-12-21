#!/usr/bin/env python

import mc1step
import mc1step_symm
import casci
import casci_symm
import addons
import casci_uhf
import mc1step_uhf
import pyscf.fci
import pyscf.scf

def CASSCF(mol, mf, *args, **kwargs):
    if mol.symmetry:
        mc = mc1step_symm.CASSCF(mol, mf, *args, **kwargs)
        mc.fcisolver = pyscf.fci.solver(mol)
    else:
        if isinstance(mf, pyscf.scf.hf.UHF) or 'UHF' in str(mf.__class__):
            mc = mc1step_uhf.CASSCF(mol, mf, *args, **kwargs)
        else:
            mc = mc1step.CASSCF(mol, mf, *args, **kwargs)
    return mc

def CASCI(mol, mf, *args, **kwargs):
    if mol.symmetry:
        mc = casci_symm.CASCI(mol, mf, *args, **kwargs)
        mc.fcisolver = pyscf.fci.solver(mol)
    else:
        if isinstance(mf, pyscf.scf.hf.UHF) or 'UHF' in str(mf.__class__):
            mc = casci_uhf.CASCI(mol, mf, *args, **kwargs)
        else:
            mc = casci.CASCI(mol, mf, *args, **kwargs)
    return mc
