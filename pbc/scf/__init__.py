#!/usr/bin/env python
# -*- coding: utf-8
# Author: Qiming Sun <osirpt.sun@gmail.com>

'''Non-relativistic and relativistic Hartree-Fock
   for periodic systems at a *single* k-point.

'''

from pyscf.pbc.scf import hf
from pyscf.pbc.scf import hf as rhf

def RHF(mol, *args):
    '''This is a wrap function to mimic pyscf 
    '''
    return rhf.RHF(mol, *args)

def KRHF(mol, *args):
    '''This is a wrap function to mimic pyscf 
    '''
    from pyscf.pbc.scf import khf
    from pyscf.pbc.scf import khf as krhf
    return krhf.KRHF(mol, *args)
