#!/usr/bin/env python
# Copyright 2014-2018 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
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
from pyscf.fci import selected_ci
select_ci = selected_ci  # for backward compatibility
from pyscf.fci import selected_ci_spin0
from pyscf.fci import selected_ci_symm
from pyscf.fci import selected_ci_spin0_symm
from pyscf.fci.selected_ci import SelectedCI, SCI

def solver(mol=None, singlet=False, symm=None):
    if mol and symm is None:
        symm = mol.symmetry
    if symm:
        if singlet:
            return direct_spin0_symm.FCISolver(mol)
        else:
            return direct_spin1_symm.FCISolver(mol)
    else:
        if singlet:
            # The code for singlet direct_spin0 sometimes gets error of 
            # "State not singlet x.xxxxxxe-06" due to numerical issues.
            # Calling direct_spin1 is slightly slower but more robust than
            # direct_spin0 especially when combining to energy penalty method
            # (:func:`fix_spin_`)
            return direct_spin0.FCISolver(mol)
        else:
            return direct_spin1.FCISolver(mol)

def FCI(mol_or_mf, mo=None, singlet=False):
    '''FCI solver

    Args:
        mol_or_mf :
            A Mole object or an SCF object

    Kwargs:
        mo :
            Molecular orbital coefficients
        singlet :
            Whether to enable spin symmetry for S=0 RHF-based FCI solver.

    Returns:
        A FCI object
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
        is_uhf = isinstance(mf, scf.uhf.UHF)
    else:
        mf = None
        mol = mol_or_mf
        is_rhf = (mo is None or (isinstance(mo, numpy.ndarray) and mo.ndim == 2))
        is_uhf = not is_rhf

    if is_uhf:
        fcisolver = direct_uhf.FCI(mol)
    else:
        fcisolver = solver(mol, singlet=(singlet and mol.spin==0))

    # Just create the FCI solver without initializing Hamiltonian
    if mo is None:
        return fcisolver

    nelec = getattr(mf, 'nelec', mol.nelec)

    if mf is None:
        hcore = scf.hf.get_hcore(mol)
        ecore = mol.energy_nuc()
    else:
        hcore = mf.get_hcore()
        ecore = mf.energy_nuc()

    if mf is None or mf._eri is None:
        if getattr(mol, 'pbc_intor', None):  # cell object has pbc_intor method
            raise NotImplementedError('Integral transformation for PBC object')
        eri_ao = mol
    else:
        eri_ao = mf._eri

    if mol.symmetry:
        if mf is None:
            s = mol.intor('int1e_ovlp')
        else:
            s = mf.get_ovlp()

        if is_uhf:
            orbsym = scf.uhf_symm.get_orbsym(mol, mo, s)
        else:
            orbsym = scf.hf_symm.get_orbsym(mol, mo, s)
    else:
        orbsym = None

    if is_uhf:
        h1e = [reduce(numpy.dot, (mo[0].conj().T, hcore, mo[0])),
               reduce(numpy.dot, (mo[1].conj().T, hcore, mo[1]))]
        eri_aa = ao2mo.kernel(eri_ao, (mo[0], mo[0], mo[0], mo[0]))
        eri_ab = ao2mo.kernel(eri_ao, (mo[0], mo[0], mo[1], mo[1]))
        eri_bb = ao2mo.kernel(eri_ao, (mo[1], mo[1], mo[1], mo[1]))
        eri = [eri_aa, eri_ab, eri_bb]
        norb = mo[0].shape[1]
    else:
        h1e = reduce(numpy.dot, (mo.conj().T, hcore, mo))
        eri = ao2mo.kernel(eri_ao, mo)
        norb = mo.shape[1]

    fcisolver_class = fcisolver.__class__
    class CISolver(fcisolver_class):
        def __init__(self, mol=None):
            fcisolver_class.__init__(self, mol)
            self.orbsym = orbsym

        def kernel(self, h1e=h1e, eri=eri, norb=norb, nelec=nelec, ci0=None,
                   ecore=ecore, **kwargs):
            return fcisolver_class.kernel(self, h1e, eri, norb, nelec, ci0,
                                          ecore=ecore, **kwargs)
    cisolver = CISolver(mol)
    cisolver.__dict__.update(fcisolver.__dict__)
    cisolver.orbsym = orbsym
    return cisolver

