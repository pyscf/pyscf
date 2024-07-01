#!/usr/bin/env python
# Copyright 2020-2023 The PySCF Developers. All Rights Reserved.
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
# Author: Hong-Zhou Ye <hzyechem@gmail.com>
#


import numpy as np

from pyscf import lib


def PipekMezey_stability_jacobi(mlo, mo_coeff=None, conv_tol=None):
    ''' Perform a stability analysis using Jacobi sweep.

        The function first identifies all pairs whose mix may lead to a loss increase greater
        than `conv_tol`. If no pairs are found, the function returns the input mo_coeff.
        Otherwise, it performs one Jacobi sweep for all pairs identified above and return the
        new mo_coeff.

        Note:
            All analysis inside this function is performed with exponent = 2, which is generally
            good enough for hopping out of local minimum.

        Args:
            mlo : PipekMezey object
            mo_coeff : np.ndarray
            conv_tol : float
                If mixing a pair of MOs leads to a loss increase greater than this parameter,
                the pair of MOs will be sent to a subsequent Jacob sweep.

        Returns:
            A tuple with two entries:

            isstable : bool
                If the input MOs are stable under Jacob rotation
            mo_coeff_new : np.ndarray
                The MOs after Jacobi rotation. If isstable is False, mo_coeff_new is the same
                as input mo_coeff.
    '''
    log = lib.logger.new_logger(mlo)

    exponent = mlo.exponent
    if exponent != 2:
        log.warn('Jacobi sweep for PipekMezey with exponent = %d is not implemented. '
                 'Exponent = 2 will be used instead.', exponent)

    if mo_coeff is None: mo_coeff = mlo.mo_coeff
    if conv_tol is None: conv_tol = mlo.conv_tol

    mol = mlo.mol
    nmo = mo_coeff.shape[1]

    if getattr(mol, 'pbc_intor', None):
        s = mol.pbc_intor('int1e_ovlp', hermi=1)
    else:
        s = mol.intor_symmetric('int1e_ovlp')

    proj = mlo.atomic_pops(mol, mo_coeff, s=s)
    pop = lib.einsum('xii->xi', proj)

    idx_tril = np.tril_indices(nmo,-1)
    def pack_tril(A):
        if A.ndim == 2:
            return A[idx_tril]
        elif A.ndim == 3:
            return np.asarray([A[i][idx_tril] for i in range(A.shape[0])])
        else:
            raise RuntimeError
    s_proj = pack_tril(proj)
    d_proj = pack_tril(lib.direct_sum('ai-aj->aij', pop, pop))*0.5
    m_proj = pack_tril(lib.direct_sum('ai+aj->aij', pop, pop))*0.5
    loss0_allpair = pack_tril(lib.direct_sum('ai+aj->ij', pop**2, pop**2))
    A_allpair = (s_proj**2 - d_proj**2.).sum(axis=0)
    B_allpair = (s_proj*d_proj).sum(axis=0)
    C_allpair = (2*m_proj**2+s_proj**2+d_proj**2).sum(axis=0)
    t_allpair = np.arctan(2*B_allpair/A_allpair)
    loss1_allpair = C_allpair - A_allpair*np.cos(4*t_allpair) - 2*B_allpair*np.sin(4*t_allpair)
    T_allpair = t_allpair + np.pi*0.25
    loss2_allpair = C_allpair - A_allpair*np.cos(4*T_allpair) - 2*B_allpair*np.sin(4*T_allpair)
    loss_allpair = np.maximum(loss1_allpair, loss2_allpair)
    dloss_allpair = loss_allpair - loss0_allpair
    mo_pair_indices = np.where(dloss_allpair > conv_tol)[0]
    dloss_pair = dloss_allpair[mo_pair_indices]
    mo_pair_indices = mo_pair_indices[np.argsort(dloss_pair)[::-1]]

    rows = np.floor(((1+8*mo_pair_indices)**0.5-1)*0.5).astype(int)
    cols = mo_pair_indices - rows*(rows+1) // 2
    rows += 1
    pairs = list(zip(rows,cols))

    if len(pairs) > 0:
        log.info('Jacobi sweep for %d mo pairs: %s', len(pairs), pairs)
        mo_coeff = _jacobi_sweep(mlo, mo_coeff, pairs, conv_tol, s=s)
        return False, mo_coeff
    else:
        return True, mo_coeff

def _jacobi_sweep(mlo, mo_coeff, mo_pairs, conv_tol, s=None):
    mol = mlo.mol
    mo_coeff = mo_coeff.copy()

    def get_loss_pair(m12s, d12s, s12s, theta):
        c2t = np.cos(2*theta)
        s2t = np.sin(2*theta)
        loss_pair = np.asarray([((m12s+d12s*c2t-s12s*s2t)**2).sum(),
                                ((m12s-d12s*c2t+s12s*s2t)**2).sum()])
        return loss_pair.sum(), loss_pair
    def rotate_pair(mo_coeff, i, j, theta):
        ct = np.cos(theta)
        st = np.sin(theta)
        mo1 = mo_coeff[:,i]*ct - mo_coeff[:,j]*st
        mo2 = mo_coeff[:,i]*st + mo_coeff[:,j]*ct
        mo_coeff[:,i] = mo1
        mo_coeff[:,j] = mo2
        return mo_coeff

    if s is None:
        if getattr(mol, 'pbc_intor', None):
            s = mol.pbc_intor('int1e_ovlp', hermi=1)
        else:
            s = mol.intor_symmetric('int1e_ovlp')

    for (idx,jdx) in mo_pairs:
        mo_pair = mo_coeff[:,[idx,jdx]]

        pop = mlo.atomic_pops(mol, mo_pair, mlo.pop_method, s=s)
        n1s = pop[:,0,0]
        n2s = pop[:,1,1]
        m12s = 0.5*(n1s + n2s)
        d12s = 0.5*(n1s - n2s)
        s12s = pop[:,0,1].real
        l0 = (n1s**2 + n2s**2).sum()

        A = (s12s**2 - d12s**2).sum()
        B = (s12s*d12s).sum()

        t1 = 0.25 * np.arctan(2*B / A)
        t2 = t1 + np.pi*0.25
        l1, l1s = get_loss_pair(m12s, d12s, s12s, t1)
        l2, l2s = get_loss_pair(m12s, d12s, s12s, t2)

        dl1 = l1 - l0
        dl2 = l2 - l0
        if dl1 > conv_tol or dl2 > conv_tol:
            (topt, ls) = (t1, l1s) if l1 > l2 else (t2, l2s)
            mo_coeff = rotate_pair(mo_coeff, idx, jdx, topt)

    return mo_coeff
