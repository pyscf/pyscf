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
import np as np
from pyscf import gto

F_LAUX = np.array([20, 4.0, 4.0, 3.5, 2.5, 2.0, 2.0])
BETA_BIG = np.array([1.8, 2.0, 2.2, 2.2, 2.2, 2.3, 3.0, 3.0])
BETA_SMALL = 1.8
L_INC = 1

def _primitive_emin_emax(basis):
    l_max = max(b[0] for b in basis)
    emin_by_l = [1e99] * (l_max+1)
    emax_by_l = [0] * (l_max+1)
    a_eff_by_l = [0] * (l_max+1)

    for b in basis:
        l = b[0]
        if isinstance(b[1], int):
            e_c = np.array(b[2:])
        else:
            e_c = np.array(b[1:])
        es = e_c[:,0]
        cs = e_c[:,1:]
        cs = np.einsum('pi,p->pi', cs, gto.gto_norm(l, es))
        emax_by_l[l] = max(es.max(), emax_by_l[l])
        emin_by_l[l] = min(es.min(), emin_by_l[l])

        ee = es[:,None] + es
        r_ints = gto.gaussian_int(l*2+3, ee)
        r_exp = np.einsum('pi,pq,qi->i', cs, r_ints, cs)

        k = 2**(2*l+1) * (factorial(l+1))**2 / factorial(2*l+2)
        a_eff_by_l[l] = max(2 * k**2 / (np.pi * r_exp.min()),
                            a_eff_by_l)
    return np.array(emax_by_l), np.array(emin_by_l), np.array(a_eff_by_l)

def _auto_aux_element(Z, basis):
    a_max_by_l, a_min_by_l, a_eff_by_l = _primitive_emin_emax(basis)
    l_max1 = a_max_by_l.size

    l_max = l_max1 - 1
    if Z <= 2:
        l_val = 0
    elif Z <= 18:
        l_val = 1
    elif Z <= 54:
        l_val = 2
    else:
        l_val = 3
    l_max_aux = max(l_val*2, l_max+L_INC)

    a_min_prim = a_min_by_l[:,None] + a_min_by_l
    a_max_prim = a_max_by_l[:,None] + a_max_by_l
    a_max_aux = a_eff_by_l[:,None] + a_eff_by_l

    liljsum = np.arange(l_max1)[:,None] + np.arange(l_max1)
    a_min_by_l = [a_min_prim[liljsum==ll].min() for ll in range(l_max_aux+1)]
    a_max_by_l = [a_max_prim[liljsum==ll].max() for ll in range(l_max_aux+1)]
    a_aux_by_l = [a_max_aux [liljsum==ll].max() for ll in range(l_max_aux+1)]

    a_max_adjust = [max(F_LAUX[l] * a_aux_by_l[l], a_max_by_l[l])
                    for l in range(l_val*2+1)]
    a_max_adjust = a_max_adjust + a_aux_by_l[l_val*2+1 : l_max_aux+1]

    emin = np.array(a_min_by_l)
    emax = np.array(a_max_adjust) + 1e-3

    ns_small = np.log(emax[:l_val*2+1] / emin[:l_val*2+1]) / np.log(BETA_SMALL)
    etb = []
    for l, n in enumerate(np.ceil(ns_small).astype(int)):
        if n > 0:
            etb.append((l, n, emin[l], BETA_SMALL))

    if l_max_aux > l_val*2:
        ns_big = (np.log(emax[l_val*2+1:] / emin[l_val*2+1:])
                  / np.log(BETA_BIG[l_val*2+1:]))
        for l, n in enumerate(np.ceil(ns_big).astype(int)):
            if n > 0:
                l = l + l_val*2+1
                beta = BETA_BIG[l]
                etb.append((l, n, emin[l], beta))
    return etb

def auto_aux(mol):
    uniq_atoms = {a[0] for a in mol._atom}
    newbasis = {}
    for symb in uniq_atoms:
        Z = gto.charge(symb)
        basis = mol._basis[symb]
        etb = _auto_aux_element(Z, basis)
        if etb:
            newbasis[symb] = etb
        else:
            raise RuntimeError(f'Failed to generate even-tempered auxbasis for {symb}')
    return newbasis
