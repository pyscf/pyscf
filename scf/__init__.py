#!/usr/bin/env python
# -*- coding: utf-8
# Author: Qiming Sun <osirpt.sun@gmail.com>

import hf
import hf_symm
import dhf
import chkfile
import addons
import diis



def RHF(mol, *args):
    if not mol.symmetry or mol.groupname is 'C1' or mol.nelectron == 1:
        return hf.RHF(mol, *args)
    else:
        return hf_symm.RHF(mol, *args)

def UHF(mol, *args):
    if not mol.symmetry or mol.groupname is 'C1' or mol.nelectron == 1:
        return hf.UHF(mol, *args)
    else:
        return hf_symm.UHF(mol, *args)

def DHF(mol, *args):
    return dhf.UHF(mol, *args)

def RKS(mol, *args):
    return dft.RKS(mol, *args)

def RDKS(mol, *args):
    return rdft.UKS(mol, *args)
