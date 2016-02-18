#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#
# FCI solver for equivalent number of alpha and beta electrons
# (requires MS=0, can be singlet, triplet, quintet, dep on init guess)
#
# Other files in the directory
# direct_spin0 singlet
# direct_spin1 arbitary number of alpha and beta electrons, based on RHF/ROHF
#              MO integrals
# direct_uhf   arbitary number of alpha and beta electrons, based on UHF
#              MO integrals
#

from pyscf.fci import cistring
from pyscf.fci import direct_spin0
from pyscf.fci import direct_spin1
from pyscf.fci import direct_uhf
from pyscf.fci import direct_spin0_symm
from pyscf.fci import direct_spin1_symm
from pyscf.fci import addons
from pyscf.fci import rdm
from pyscf.fci import spin_op
from pyscf.fci.cistring import num_strings
from pyscf.fci.rdm import reorder_rdm
from pyscf.fci.spin_op import spin_square
from pyscf.fci.direct_spin1 import make_pspace_precond, make_diag_precond
from pyscf.fci import direct_nosym

def solver(mol, singlet=True, symm=None):
    if symm is None:
        symm = mol.symmetry
    if symm:
        if singlet:
            return direct_spin0_symm.FCISolver(mol)
        else:
            return direct_spin1_symm.FCISolver(mol)
    else:
        if singlet:
            return direct_spin0.FCISolver(mol)
        else:
            return direct_spin1.FCISolver(mol)

def FCI(mol, mo, singlet=True):
    from functools import reduce
    import numpy
    from pyscf import scf
    from pyscf import symm
    from pyscf import ao2mo
    if mol.spin > 0:
        cis = solver(mol, False)
    else:
        cis = solver(mol, singlet)
    class CISolver(cis.__class__):
        def __init__(self):
            self.__dict__.update(cis.__dict__)
            self.h1e = reduce(numpy.dot, (mo.T, scf.hf.get_hcore(mol), mo))
            self.eri = ao2mo.full(mol, mo)
            self.eci = None
            self.ci = None
            if mol.symmetry:
                self.orbsym = symm.label_orb_symm(mol, mol.irrep_id,
                                                  mol.symm_orb, mo)
            self._keys = set(self.__dict__.keys())

        def kernel(self, h1e=None, eri=None, norb=None, nelec=None, ci0=None, **kwargs):
            if h1e is None: h1e = self.h1e
            if eri is None: eri = self.eri
            if norb is None: norb = mo.shape[1]
            if nelec is None:
                nelec_a = (mol.nelectron + mol.spin) // 2
                nelec_b = mol.nelectron - nelec_a
                nelec = (nelec_a, nelec_b)
            self.eci, self.ci = \
                    cis.__class__.kernel(self, h1e, eri, norb, nelec, ci0, **kwargs)
            return self.eci, self.ci
    return CISolver()

