#!/usr/bin/env python
# -*- coding: utf-8
# Author: Qiming Sun <osirpt.sun@gmail.com>

from pyscf.scf import hf
from pyscf.scf import hf_symm
from pyscf.scf import dhf
from pyscf.scf import chkfile
from pyscf.scf import diis
from pyscf.scf import addons
from pyscf.scf import dfhf


def RHF(mol, *args):
    if mol.nelectron == 1:
        return hf.HF1e(mol)
    elif not mol.symmetry or mol.groupname is 'C1':
        if mol.spin > 0:
            return hf.ROHF(mol, *args)
        else:
            return hf.RHF(mol, *args)
    else:
        if mol.spin > 0:
            return hf_symm.ROHF(mol, *args)
        else:
            return hf_symm.RHF(mol, *args)

def ROHF(mol, *args):
    if mol.nelectron == 1:
        return hf.HF1e(mol)
    elif not mol.symmetry or mol.groupname is 'C1':
        return hf.ROHF(mol, *args)
    else:
        return hf_symm.ROHF(mol, *args)

def UHF(mol, *args):
    if mol.nelectron == 1:
        return hf.HF1e(mol)
    elif not mol.symmetry or mol.groupname is 'C1':
        return hf.UHF(mol, *args)
    else:
        return hf_symm.UHF(mol, *args)

def DHF(mol, *args):
    if mol.nelectron == 1:
        return dhf.HF1e(mol)
    else:
        return dhf.UHF(mol, *args)



