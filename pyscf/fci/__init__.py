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
from pyscf.fci.select_ci import SelectedCI, SCI

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

def FCI(mol_or_mf, mo=None, singlet=True):
    '''FCI solver
    '''
    from functools import reduce
    import numpy
    from pyscf import scf
    from pyscf import symm
    from pyscf import ao2mo
    if isinstance(mol_or_mf, scf.hf.SCF):
        mf = mol_or_mf
        mol = mf.mol
        if mo is None:
            mo = mf.mo_coeff
    else:
        mf = None
        mol = mol_or_mf
    cis = solver(mol, singlet=(singlet and mol.spin==0))
    if mo is None:
        return cis

    class CISolver(cis.__class__):
        def __init__(self):
            self.__dict__.update(cis.__dict__)
            self.orbsym = None
            self.eci = None
            self.ci = None
            if mol.symmetry:
                if mf is None:
                    self.orbsym = scf.hf_symm.get_orbsym(mol, mo)
                else:
                    self.orbsym = scf.hf_symm.get_orbsym(mol, mo, mf.get_ovlp(mol))
            self._keys = set(self.__dict__.keys())

        def kernel(self, h1e=None, eri=None, norb=None, nelec=None, ci0=None,
                   ecore=None, **kwargs):
            if h1e is None or eri is None:
                if mf is None:
                    if h1e is None:
                        h1e = reduce(numpy.dot, (mo.T, scf.hf.get_hcore(mol), mo))
                    if eri is None:
                        eri = ao2mo.full(mol, mo)
                    if ecore is None:
                        ecore = mol.energy_nuc()
                else:
                    if h1e is None:
                        h1e = reduce(numpy.dot, (mo.T, mf.get_hcore(mol), mo))
                    if eri is None:
                        if mf._eri is None:
                            eri = ao2mo.full(mol, mo)
                        else:
                            eri = ao2mo.full(mf._eri, mo)
                    if ecore is None:
                        ecore = mf.energy_nuc()
            if norb is None: norb = mo.shape[1]
            if nelec is None: nelec = mol.nelec
            self.eci, self.ci = \
                    cis.__class__.kernel(self, h1e, eri, norb, nelec, ci0,
                                         ecore=ecore, **kwargs)
            return self.eci, self.ci
    return CISolver()

