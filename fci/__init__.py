#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
Different FCI solvers are implemented to support different type of symmetry.
                    Symmetry
File                Point group   Spin singlet   Real hermitian*    Alpha/beta degeneracy
direct_spin0_symm   Yes           Yes            Yes                Yes
direct_spin1_symm   Yes           No             Yes                Yes
direct_spin0        No            Yes            Yes                Yes
direct_spin1        No            No             Yes                Yes
direct_uhf          No            No             Yes                No
direct_nosym        No            No             No**               Yes

*  Real hermitian Hamiltonian implies (ij|kl) = (ji|kl) = (ij|lk) = (ji|lk)
** Hamiltonian is real but not hermitian, (ij|kl) != (ji|kl) ...
'''

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
from pyscf.fci import select_ci
from pyscf.fci import select_ci_spin0
from pyscf.fci import select_ci_symm
from pyscf.fci import select_ci_spin0_symm
from pyscf.fci.select_ci import SelectCI, SCI

def solver(mol=None, singlet=True, symm=None):
    if mol and symm is None:
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
    '''FCI solver

    Pass nelec to kernel
    '''
    from functools import reduce
    import numpy
    from pyscf import scf
    from pyscf import symm
    from pyscf import ao2mo
    cis = solver(mol, singlet=(mol.spin==0))
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

        def kernel(self, h1e=None, eri=None, norb=None, nelec=None, ci0=None,
                   ecore=None, **kwargs):
            if h1e is None: h1e = self.h1e
            if eri is None: eri = self.eri
            if norb is None: norb = mo.shape[1]
            if nelec is None: nelec = mol.nelec
            if ecore is None: ecore = mol.energy_nuc()
            self.eci, self.ci = \
                    cis.__class__.kernel(self, h1e, eri, norb, nelec, ci0,
                                         ecore=ecore, **kwargs)
            return self.eci, self.ci
    return CISolver()

