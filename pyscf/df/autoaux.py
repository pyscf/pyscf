#!/usr/bin/env python
# Copyright 2024 The PySCF Developers. All Rights Reserved.
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

'''
The AutoAux algorithm by ORCA

Ref:
    JCTC, 13 (2016), 554
'''

from math import factorial
import numpy as np
from pyscf import gto
from pyscf.lib import logger

F_LAUX   = np.array([20 , 7.0, 4.0, 4.0, 3.5, 2.5, 2.0, 2.0])
BETA_BIG = np.array([1.8, 2.0, 2.2, 2.2, 2.2, 2.3, 3.0, 3.0])
BETA_SMALL = 1.8

def _primitive_emin_emax(basis):
    l_max = max(b[0] for b in basis)
    emin_by_l = [1e99] * (l_max+1)
    emax_by_l = [0] * (l_max+1)
    e_eff_by_l = [0] * (l_max+1)

    for b in basis:
        l = b[0]
        if isinstance(b[1], (int, np.integer)):
            e_c = np.array(b[2:])
        else:
            e_c = np.array(b[1:])
        es = e_c[:,0]
        emax_by_l[l] = max(es.max(), emax_by_l[l])
        emin_by_l[l] = min(es.min(), emin_by_l[l])

        cs = e_c[:,1:]
        cs = np.einsum('pi,p->pi', cs, gto.gto_norm(l, es)) # normalize GTOs
        cs = gto.mole._nomalize_contracted_ao(l, es, cs) # prefactors in r_ints
        ee = es[:,None] + es
        r_ints = gto.gaussian_int(l*2+3, ee) # \int \chi^2 r dr
        r_exp = np.einsum('pi,pq,qi->i', cs, r_ints, cs)

        k = 2**(2*l+1) * factorial(l+1)**2 / factorial(2*l+2)
        # Eq (9) in the paper, e_eff = 2 * k**2 / (np.pi * r_exp) is a typo.
        # See also https://github.com/MolSSI-BSE/basis_set_exchange/issues/317
        # For primitive functions, following expression leads to
        # e_eff = exponent of the basis
        e_eff = 2 * k**2 / (np.pi * r_exp**2)
        # For primitive functions, e_eff may be slightly different to the
        # exponent due to the rounding errors in gaussian_int function.
        # When a particular shell has only one primitive function, one auxiliary
        # function should be generated. This error can introduce an additional
        # auxiliary function.
        # Slightly reduce e_eff to remove the extra auxiliary functions.
        e_eff -= 1e-8
        e_eff_by_l[l] = max(e_eff.max(), e_eff_by_l[l])
    return np.array(emax_by_l), np.array(emin_by_l), np.array(e_eff_by_l)

def _auto_aux_element(Z, basis, ecp_core=0):
    a_max_by_l, a_min_by_l, a_eff_by_l = _primitive_emin_emax(basis)
    a_min_prim = a_min_by_l[:,None] + a_min_by_l
    a_max_prim = a_max_by_l[:,None] + a_max_by_l
    a_max_aux = a_eff_by_l[:,None] + a_eff_by_l

    l_max1 = a_max_by_l.size
    l_max = l_max1 - 1
    # TODO: handle ECP
    if Z <= 2:
        l_val = 0
    elif Z <= 20:
        l_val = 1
    elif Z <= 56:
        l_val = 2
    else:
        l_val = 3
    l_inc = 1
    if Z > 18:
        l_inc = 2
    l_max_aux = min(max(l_val*2, l_max+l_inc), l_max*2)

    liljsum = np.arange(l_max1)[:,None] + np.arange(l_max1)
    liljsub = abs(np.arange(l_max1)[:,None] - np.arange(l_max1))
    a_min_by_l = [a_min_prim[(liljsub<=ll) & (ll<=liljsum)].min() for ll in range(l_max_aux+1)]
    a_max_by_l = [a_max_prim[(liljsub<=ll) & (ll<=liljsum)].max() for ll in range(l_max_aux+1)]
    a_aux_by_l = [a_max_aux [(liljsub<=ll) & (ll<=liljsum)].max() for ll in range(l_max_aux+1)]

    a_max_adjust = [min(F_LAUX[l] * a_aux_by_l[l], a_max_by_l[l])
                    for l in range(l_val*2+1)]
    a_max_adjust = a_max_adjust + a_aux_by_l[l_val*2+1 : l_max_aux+1]

    emin = np.array(a_min_by_l)
    emax = np.array(a_max_adjust)

    ns_small = np.log(a_max_adjust[:l_val*2+1] / emin[:l_val*2+1]) / np.log(BETA_SMALL)
    etb = []
    # ns_small+1 to ensure the largest exponent in etb > emax
    for l, n in enumerate(np.ceil(ns_small).astype(int) + 1):
        if n > 0:
            etb.append((l, n, emin[l], BETA_SMALL))

    if l_max_aux > l_val*2:
        ns_big = (np.log(emax[l_val*2+1:] / emin[l_val*2+1:])
                  / np.log(BETA_BIG[l_val*2+1:l_max_aux+1]))
        for l, n in enumerate(np.ceil(ns_big).astype(int) + 1):
            if n > 0:
                l = l + l_val*2+1
                beta = BETA_BIG[l]
                etb.append((l, n, emin[l], beta))
    return etb

def autoaux(mol):
    '''
    Create an auxiliary basis set for the given orbital basis set using
    the Auto-Aux algorithm.

    See also: G. L. Stoychev, A. A. Auer, and F. Neese
    Automatic Generation of Auxiliary Basis Sets
    J. Chem. Theory Comput. 13, 554 (2017)
    http://doi.org/10.1021/acs.jctc.6b01041
    '''
    from pyscf.gto.basis import bse

    def expand(symb):
        Z = gto.charge(symb)
        etb = _auto_aux_element(Z, mol._basis[symb])
        if etb:
            for l, n, emin, beta in etb:
                logger.info(mol, 'ETB for %s: l = %d, exps = %s * %g^n , n = 0..%d',
                            symb, l, emin, beta, n-1)
            return gto.expand_etbs(etb)
        raise RuntimeError(f'Failed to generate even-tempered auxbasis for {symb}')

    uniq_atoms = {a[0] for a in mol._atom}
    if bse.basis_set_exchange is None:
        return {symb: expand(symb) for symb in uniq_atoms}

    if isinstance(mol.basis, str):
        try:
            elements = [gto.charge(symb) for symb in uniq_atoms]
            newbasis = bse.autoaux(mol.basis, elements)
        except KeyError:
            newbasis = {symb: expand(symb) for symb in uniq_atoms}
    else:
        newbasis = {}
        for symb in uniq_atoms:
            if symb in mol.basis and isinstance(mol.basis[symb], str):
                try:
                    auxbs = bse.autoaux(mol.basis[symb], gto.charge(symb))
                    newbasis[symb] = next(iter(auxbs.values()))
                except KeyError:
                    newbasis[symb] = expand(symb)
            else:
                newbasis[symb] = expand(symb)
    return newbasis

def autoabs(mol):
    '''
    Create a Coulomb fitting basis set for the given orbital basis set.
    See also:
    R. Yang, A. P. Rendell, and M. J. Frisch
    Automatically generated Coulomb fitting basis sets: Design and accuracy for systems containing H to Kr
    J. Chem. Phys. 127, 074102 (2007)
    http://doi.org/10.1063/1.2752807
    '''
    from pyscf.gto.basis import bse
    if bse is None:
        print('Package basis-set-exchange not available')
        raise ImportError

    uniq_atoms = {a[0] for a in mol._atom}
    if isinstance(mol.basis, str):
        elements = [gto.charge(symb) for symb in uniq_atoms]
        newbasis = bse.autoabs(mol.basis, elements)
    else:
        newbasis = {}
        for symb in uniq_atoms:
            if symb in mol.basis and isinstance(mol.basis[symb], str):
                auxbs = bse.autoabs(mol.basis[symb], gto.charge(symb))
                newbasis[symb] = next(iter(auxbs.values()))
            else:
                raise NotImplementedError
    return newbasis
