#!/usr/bin/env python
# -*- coding: utf-8
# Author: Qiming Sun <osirpt.sun@gmail.com>

from pyscf.scf import hf
from pyscf.scf import hf as rhf
from pyscf.scf import hf_symm
from pyscf.scf import hf_symm as rhf_symm
from pyscf.scf import uhf
from pyscf.scf import uhf_symm
from pyscf.scf import dhf
from pyscf.scf import chkfile
from pyscf.scf import diis
from pyscf.scf import addons
from pyscf.scf.dfhf import density_fit
from pyscf.scf.uhf import spin_square
from pyscf.scf.hf import get_init_guess
from pyscf.scf.addons import *


def RHF(mol, *args):
    if mol.nelectron == 1:
        return rhf.HF1e(mol)
    elif not mol.symmetry or mol.groupname is 'C1':
        if mol.spin > 0:
            return rhf.ROHF(mol, *args)
        else:
            return rhf.RHF(mol, *args)
    else:
        if mol.spin > 0:
            return rhf_symm.ROHF(mol, *args)
        else:
            return rhf_symm.RHF(mol, *args)

def ROHF(mol, *args):
    if mol.nelectron == 1:
        return rhf.HF1e(mol)
    elif not mol.symmetry or mol.groupname is 'C1':
        return rhf.ROHF(mol, *args)
    else:
        return hf_symm.ROHF(mol, *args)

def UHF(mol, *args):
    if mol.nelectron == 1:
        return rhf.HF1e(mol)
    elif not mol.symmetry or mol.groupname is 'C1':
        return uhf.UHF(mol, *args)
    else:
        return uhf_symm.UHF(mol, *args)

def DHF(mol, *args):
    if mol.nelectron == 1:
        return dhf.HF1e(mol)
    else:
        return dhf.UHF(mol, *args)


