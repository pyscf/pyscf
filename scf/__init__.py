#!/usr/bin/env python
# -*- coding: utf-8
# Author: Qiming Sun <osirpt.sun@gmail.com>

import os, sys
import diis
import hf
import hf_symm
from hf import SCF
from chkfile import dump_chkfile_key, load_chkfile_key
#import dhf
#import dft
#import rdft

#import dhf_dkb
#import atom_hf
import molden_dump


def RHF(mol, *args):
    if not mol.symmetry or mol.pgname is 'C1' or mol.nelectron == 1:
        return hf.RHF(mol, *args)
    else:
        return hf_symm.RHF(mol, *args)

def UHF(mol, *args):
    if not mol.symmetry or mol.pgname is 'C1' or mol.nelectron == 1:
        return hf.UHF(mol, *args)
    else:
        return hf_symm.UHF(mol, *args)

def DHF(mol, *args):
    return dhf.UHF(mol, *args)

def RKS(mol, *args):
    return dft.RKS(mol, *args)

def RDKS(mol, *args):
    return rdft.UKS(mol, *args)
