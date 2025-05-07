#!/usr/bin/env python
#
# This code was copied from the data generation program of Tencent Alchemy
# project (https://github.com/tencent-alchemy).
#

#
# Copyright 2019 Tencent America LLC. All Rights Reserved.
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
Non-relativistic RHF analytical Hessian with density-fitting approximation

Ref:
[1] Efficient implementation of the analytic second derivatives of
    Hartree-Fock and hybrid DFT energies: a detailed analysis of different
    approximations.  Dmytro Bykov, Taras Petrenko, Robert Izsak, Simone
    Kossmann, Ute Becker, Edward Valeev, Frank Neese. Mol. Phys. 113, 1961 (2015)
'''


import numpy
import numpy as np
import scipy.linalg
from pyscf import lib
from pyscf.lib import logger
from pyscf import ao2mo
from pyscf import df
from pyscf.hessian import rhf as rhf_hess
from pyscf.df.grad.rhf import (_int3c_wrapper, _gen_metric_solver,
                               LINEAR_DEP_THRESHOLD)

def _pinv(a, lindep=LINEAR_DEP_THRESHOLD):
    '''Similar to pinv (v1.7.0) with atol=lindep and rtol=0'''
    w, v = scipy.linalg.eigh(a)
    mask = w > lindep
    v1 = v[:, mask]
    return lib.dot(v1/w[mask], v1.conj().T)

def partial_hess_elec(hessobj, mo_energy=None, mo_coeff=None, mo_occ=None,
                      atmlst=None, max_memory=4000, verbose=None):
    e1, ej, ek = _partial_hess_ejk(hessobj, mo_energy, mo_coeff, mo_occ,
                                   atmlst, max_memory, verbose, True)
    return e1 + ej - ek

def _partial_hess_ejk(hessobj, mo_energy=None, mo_coeff=None, mo_occ=None,
                      atmlst=None, max_memory=4000, verbose=None, with_k=True):
    '''Partial derivative
    '''
    log = logger.new_logger(hessobj, verbose)
    time0 = t1 = (logger.process_clock(), logger.perf_counter())

    mol = hessobj.mol
    mf = hessobj.base
    if mo_energy is None: mo_energy = mf.mo_energy
    if mo_occ is None:    mo_occ = mf.mo_occ
    if mo_coeff is None:  mo_coeff = mf.mo_coeff
    if atmlst is None: atmlst = range(mol.natm)

    nao, nmo = mo_coeff.shape
    mocc = mo_coeff[:,mo_occ>0]
    mocc_2 = np.einsum('pi,i->pi', mocc, mo_occ[mo_occ>0]**.5)
    nocc = mocc.shape[1]
    dm0 = numpy.dot(mocc, mocc.T) * 2
    # Energy weighted density matrix
    dme0 = numpy.einsum('pi,qi,i->pq', mocc, mocc, mo_energy[mo_occ>0]) * 2

    with_df = hessobj.base.with_df
    auxmol = with_df.auxmol
    if auxmol is None:
        auxmol = df.addons.make_auxmol(with_df.mol, with_df.auxbasis)
    naux = auxmol.nao
    nbas = mol.nbas
    auxslices = auxmol.aoslice_by_atom()
    aoslices = mol.aoslice_by_atom()
    aux_loc = auxmol.ao_loc
    blksize = min(480, hessobj.max_memory*.3e6/8/nao**2)
    aux_ranges = ao2mo.outcore.balance_partition(auxmol.ao_loc, blksize)

    hcore_deriv = hessobj.hcore_generator(mol)
    s1aa, s1ab, s1a = rhf_hess.get_ovlp(mol)

    ftmp = lib.H5TmpFile()
    get_int3c = _int3c_wrapper(mol, auxmol, 'int3c2e', 's1')
    # Without RI basis response
    #    (20|0)(0|00)
    #    (11|0)(0|00)
    #    (10|0)(0|10)
    int2c = auxmol.intor('int2c2e', aosym='s1')
    solve_j2c = _gen_metric_solver(int2c)
    int2c_ip1 = auxmol.intor('int2c2e_ip1', aosym='s1')

    rhoj0_P = 0
    if hessobj.max_memory*.8e6/8 < naux*nocc*(nocc+nao):
        raise RuntimeError('Memory not enough. You need to increase mol.max_memory')
    rhok0_Pl_ = np.empty((naux,nao,nocc))
    for i, (shl0, shl1, p0, p1) in enumerate(aoslices):
        int3c = get_int3c((shl0, shl1, 0, nbas, 0, auxmol.nbas))
        rhoj0_P += np.einsum('klp,kl->p', int3c, dm0[p0:p1])
        tmp = lib.einsum('ijp,jk->pik', int3c, mocc_2)
        tmp = solve_j2c(tmp.reshape(naux,-1))
        rhok0_Pl_[:,p0:p1] = tmp.reshape(naux,p1-p0,nocc)
        int3c = tmp = None
    rhoj0_P = solve_j2c(rhoj0_P)

    get_int3c_ipip1 = _int3c_wrapper(mol, auxmol, 'int3c2e_ipip1', 's1')
    vj1_diag = 0
    vk1_diag = 0
    for shl0, shl1, nL in aux_ranges:
        shls_slice = (0, nbas, 0, nbas, shl0, shl1)
        p0, p1 = aux_loc[shl0], aux_loc[shl1]
        int3c_ipip1 = get_int3c_ipip1(shls_slice)
        vj1_diag += np.einsum('xijp,p->xij', int3c_ipip1, rhoj0_P[p0:p1]).reshape(3,3,nao,nao)
        if with_k:
            tmp = lib.einsum('Plj,Jj->PlJ', rhok0_Pl_[p0:p1], mocc_2)
            vk1_diag += lib.einsum('xijp,plj->xil', int3c_ipip1, tmp).reshape(3,3,nao,nao)
    int3c_ipip1 = get_int3c_ipip1 = tmp = None
    t1 = log.timer_debug1('contracting int2e_ipip1', *t1)

    get_int3c_ip1 = _int3c_wrapper(mol, auxmol, 'int3c2e_ip1', 's1')
    rho_ip1 = ftmp.create_dataset('rho_ip1', (nao,nao,naux,3), 'f8')
    rhok_ip1_IkP = ftmp.create_group('rhok_ip1_IkP')
    rhok_ip1_PkI = ftmp.create_group('rhok_ip1_PkI')
    rhoj1 = np.empty((mol.natm,naux,3))
    wj1 = np.empty((mol.natm,naux,3))
    for i0, ia in enumerate(atmlst):
        shl0, shl1, p0, p1 = aoslices[ia]
        shls_slice = (shl0, shl1, 0, nbas, 0, auxmol.nbas)
        int3c_ip1 = get_int3c_ip1(shls_slice)
        tmp_ip1 = solve_j2c(int3c_ip1.reshape(-1,naux).T).reshape(naux,3,p1-p0,nao)
        rhoj1[i0] = np.einsum('pxij,ji->px', tmp_ip1, dm0[:,p0:p1])
        wj1[i0] = np.einsum('xijp,ji->px', int3c_ip1, dm0[:,p0:p1])
        rho_ip1[p0:p1] = tmp_ip1.transpose(2,3,0,1)
        if with_k:
            tmp = lib.einsum('pykl,li->ikpy', tmp_ip1, dm0)
            rhok_ip1_IkP['%.4d'%ia] = tmp
            rhok_ip1_PkI['%.4d'%ia] = tmp.transpose(2,1,0,3)
            tmp = None
    ej = lib.einsum('ipx,jpy->ijxy', rhoj1, wj1) * 4
    ek = np.zeros_like(ej)
    e1 = np.zeros_like(ej)
    rhoj1 = wj1 = None

    if with_k:
        vk2buf = 0
        for shl0, shl1, nL in aux_ranges:
            shls_slice = (0, nbas, 0, nbas, shl0, shl1)
            p0, p1 = aux_loc[shl0], aux_loc[shl1]
            int3c_ip1 = get_int3c_ip1(shls_slice)
            vk2buf += lib.einsum('xijp,pkjy->xyki', int3c_ip1,
                                 _load_dim0(rhok_ip1_PkI, p0, p1))
            int3c_ip1 = None

    get_int3c_ip2 = _int3c_wrapper(mol, auxmol, 'int3c2e_ip2', 's1')
    wj_ip2 = np.empty((naux,3))
    wk_ip2_Ipk = ftmp.create_dataset('wk_ip2', (nao,naux,3,nao), 'f8')
    if hessobj.auxbasis_response > 1:
        wk_ip2_P__ = np.empty((naux,3,nocc,nocc))
    for shl0, shl1, nL in aux_ranges:
        shls_slice = (0, nbas, 0, nbas, shl0, shl1)
        p0, p1 = aux_loc[shl0], aux_loc[shl1]
        int3c_ip2 = get_int3c_ip2(shls_slice)
        wj_ip2[p0:p1] = np.einsum('yklp,lk->py', int3c_ip2, dm0)
        if with_k:
            wk_ip2_Ipk[:,p0:p1] = lib.einsum('yklp,il->ipyk', int3c_ip2, dm0)
            if hessobj.auxbasis_response > 1:
                wk_ip2_P__[p0:p1] = lib.einsum('xuvp,ui,vj->pxij', int3c_ip2, mocc_2, mocc_2)
        int3c_ip2 = None

    if hessobj.auxbasis_response > 1:
        get_int3c_ipip2 = _int3c_wrapper(mol, auxmol, 'int3c2e_ipip2', 's1')
        rhok0_P__ = lib.einsum('plj,li->pij', rhok0_Pl_, mocc_2)
        rho2c_0 = lib.einsum('pij,qji->pq', rhok0_P__, rhok0_P__)
        int2c_inv = _pinv(int2c, lindep=LINEAR_DEP_THRESHOLD)
        int2c_ipip1 = auxmol.intor('int2c2e_ipip1', aosym='s1')
        int2c_ip_ip  = lib.einsum('xpq,qr,ysr->xyps', int2c_ip1, int2c_inv, int2c_ip1)
        int2c_ip_ip -= auxmol.intor('int2c2e_ip1ip2', aosym='s1').reshape(3,3,naux,naux)
    int2c = solve_j2c = None

    get_int3c_ipvip1 = _int3c_wrapper(mol, auxmol, 'int3c2e_ipvip1', 's1')
    get_int3c_ip1ip2 = _int3c_wrapper(mol, auxmol, 'int3c2e_ip1ip2', 's1')

    for i0, ia in enumerate(atmlst):
        shl0, shl1, p0, p1 = aoslices[ia]
        shls_slice = (shl0, shl1, 0, nbas, 0, auxmol.nbas)
        # (10|0)(0|10) without response of RI basis
        if with_k:
            int3c_ip1 = get_int3c_ip1(shls_slice)
            vk1 = lib.einsum('xijp,ikpy->xykj', int3c_ip1, _load_dim0(rhok_ip1_IkP, p0, p1))
            vk1[:,:,:,p0:p1] += vk2buf[:,:,:,p0:p1]
        t1 = log.timer_debug1('contracting int2e_ip1ip2 for atom %d'%ia, *t1)
        int3c_ip1 = None

        # (11|0)(0|00) without response of RI basis
        int3c_ipvip1 = get_int3c_ipvip1(shls_slice)
        vj1 = np.einsum('xijp,p->xji', int3c_ipvip1, rhoj0_P).reshape(3,3,nao,p1-p0)
        if with_k:
            tmp = lib.einsum('pki,ji->pkj', rhok0_Pl_, mocc_2[p0:p1])
            vk1 += lib.einsum('xijp,pki->xjk', int3c_ipvip1, tmp).reshape(3,3,nao,nao)
        t1 = log.timer_debug1('contracting int2e_ipvip1 for atom %d'%ia, *t1)
        int3c_ipvip1 = tmp = None

        e1[i0,i0] -= numpy.einsum('xypq,pq->xy', s1aa[:,:,p0:p1], dme0[p0:p1])*2
        ej[i0,i0] += numpy.einsum('xypq,pq->xy', vj1_diag[:,:,p0:p1], dm0[p0:p1])*2
        if with_k:
            ek[i0,i0] += numpy.einsum('xypq,pq->xy', vk1_diag[:,:,p0:p1], dm0[p0:p1])

        for j0, ja in enumerate(atmlst[:i0+1]):
            q0, q1 = aoslices[ja][2:]
            ej[i0,j0] += numpy.einsum('xypq,pq->xy', vj1[:,:,q0:q1], dm0[q0:q1,p0:p1])*2
            e1[i0,j0] -= numpy.einsum('xypq,pq->xy', s1ab[:,:,p0:p1,q0:q1], dme0[p0:p1,q0:q1])*2
            if with_k:
                ek[i0,j0] += numpy.einsum('xypq,pq->xy', vk1[:,:,q0:q1], dm0[q0:q1])

            h1ao = hcore_deriv(ia, ja)
            e1[i0,j0] += numpy.einsum('xypq,pq->xy', h1ao, dm0)

        #
        # The first order RI basis response
        #
        #    (10|1)(0|00)
        #    (10|0)(1|0)(0|00)
        #    (10|0)(0|1)(0|00)
        #    (10|0)(1|00)
        if hessobj.auxbasis_response:
            wk1_Pij = rho_ip1[p0:p1].transpose(2,3,0,1)
            rhoj1_P = np.einsum('pxij,ji->px', wk1_Pij, dm0[:,p0:p1])
            # (10|1)(0|0)(0|00)
            int3c_ip1ip2 = get_int3c_ip1ip2(shls_slice)
            wj11_p = np.einsum('xijp,ji->xp', int3c_ip1ip2, dm0[:,p0:p1])
            # (10|0)(1|0)(0|00)
            wj0_01 = np.einsum('ypq,q->yp', int2c_ip1, rhoj0_P)
            if with_k:
                rhok0_P_I = lib.einsum('plj,il->pji', rhok0_Pl_, dm0[p0:p1])
                rhok0_PJI = lib.einsum('pji,Jj->pJi', rhok0_P_I, mocc_2)
                wk1_pJI = lib.einsum('ypq,qji->ypji', int2c_ip1, rhok0_PJI)
                wk1_IpJ = lib.einsum('ipyk,kj->ipyj', wk_ip2_Ipk[p0:p1], dm0)
                #rho2c_PQ = lib.einsum('qij,uj,iupx->xqp', rhok0_Pl_, mocc_2[p0:p1], rhok_ip1_IkP['%.4d'%ia])
                rho2c_PQ = lib.einsum('pxij,qji->xqp', wk1_Pij, rhok0_PJI)
            for j0, (q0, q1) in enumerate(auxslices[:,2:]):
                # (10|1)(0|00)
                _ej  = np.einsum('xp,p->x', wj11_p[:,q0:q1], rhoj0_P[q0:q1]).reshape(3,3)
                # (10|0)(0|1)(0|00)
                _ej -= lib.einsum('yqp,q,px->xy', int2c_ip1[:,q0:q1], rhoj0_P[q0:q1], rhoj1_P)
                # (10|0)(1|0)(0|00)
                _ej -= lib.einsum('px,yp->xy', rhoj1_P[q0:q1], wj0_01[:,q0:q1])
                # (10|0)(1|00)
                _ej += lib.einsum('px,py->xy', rhoj1_P[q0:q1], wj_ip2[q0:q1])
                if hessobj.auxbasis_response > 1:
                    ej[i0,j0] += _ej * 2
                    ej[j0,i0] += _ej.T * 2
                else:
                    ej[i0,j0] += _ej
                    ej[j0,i0] += _ej.T
                if with_k:
                    _ek  = lib.einsum('xijp,pji->x', int3c_ip1ip2[:,:,:,q0:q1],
                                      rhok0_PJI[q0:q1]).reshape(3,3)
                    _ek -= lib.einsum('pxij,ypji->xy', wk1_Pij[q0:q1], wk1_pJI[:,q0:q1])
                    _ek -= lib.einsum('xqp,yqp->xy', rho2c_PQ[:,q0:q1], int2c_ip1[:,q0:q1])
                    _ek += lib.einsum('pxij,ipyj->xy', wk1_Pij[q0:q1], wk1_IpJ[:,q0:q1])
                    if hessobj.auxbasis_response > 1:
                        ek[i0,j0] += _ek
                        ek[j0,i0] += _ek.T
                    else:
                        ek[i0,j0] += _ek * .5
                        ek[j0,i0] += _ek.T * .5
            int3c_ip1ip2 = rhok0_P_I = rhok0_PJI = wk1_pJI = wk1_IpJ = rho2c_PQ = None

        #
        # The second order RI basis response
        #
        if hessobj.auxbasis_response > 1:
            # (00|2)(0|00)
            # (00|0)(2|0)(0|00)
            shl0, shl1, p0, p1 = auxslices[ia]
            shls_slice = (0, nbas, 0, nbas, shl0, shl1)
            int3c_ipip2 = get_int3c_ipip2(shls_slice)
            ej[i0,i0] += np.einsum('xijp,ji,p->x', int3c_ipip2, dm0, rhoj0_P[p0:p1]).reshape(3,3)
            ej[i0,i0] -= np.einsum('p,xpq,q->x', rhoj0_P[p0:p1], int2c_ipip1[:,p0:p1], rhoj0_P).reshape(3,3)

            if with_k:
                rhok0_PJI = lib.einsum('Pij,Jj,Ii->PJI', rhok0_P__[p0:p1], mocc_2, mocc_2)
                ek[i0,i0] += .5 * np.einsum('xijp,pij->x', int3c_ipip2, rhok0_PJI).reshape(3,3)
                ek[i0,i0] -= .5 * np.einsum('pq,xpq->x', rho2c_0[p0:p1], int2c_ipip1[:,p0:p1]).reshape(3,3)
                rhok0_PJI = None
            # (00|0)(1|1)(0|00)
            # (00|1)(1|0)(0|00)
            # (00|1)(0|1)(0|00)
            # (00|1)(1|00)
            rhoj1 = lib.einsum('px,pq->xq', wj_ip2[p0:p1], int2c_inv[p0:p1])
            # (00|0)(0|1)(1|0)(0|00)
            rhoj0_01 = lib.einsum('xp,pq->xq', wj0_01[:,p0:p1], int2c_inv[p0:p1])
            # (00|0)(1|0)(1|0)(0|00)
            ip1_2c_2c = lib.einsum('xpq,qr->xpr', int2c_ip1[:,p0:p1], int2c_inv)
            rhoj0_10 = lib.einsum('p,xpq->xq', rhoj0_P[p0:p1], ip1_2c_2c)
            if with_k:
                # (00|0)(0|1)(1|0)(0|00)
                ip1_rho2c = .5 * lib.einsum('xpq,qr->xpr', int2c_ip1[:,p0:p1], rho2c_0)
                rho2c_1  = lib.einsum('xrq,rp->xpq', ip1_rho2c, int2c_inv[p0:p1])
                # (00|0)(1|0)(1|0)(0|00)
                rho2c_1 += lib.einsum('xrp,rq->xpq', ip1_2c_2c, rho2c_0[p0:p1])
                # (00|1)(0|1)(0|00)
                # (00|1)(1|0)(0|00)
                int3c_ip2 = get_int3c_ip2(shls_slice)
                tmp = lib.einsum('xuvr,vj,ui->xrij', int3c_ip2, mocc_2, mocc_2)
                tmp = lib.einsum('xrij,qij,rp->xpq', tmp, rhok0_P__, int2c_inv[p0:p1])
                rho2c_1 -= tmp
                rho2c_1 -= tmp.transpose(0,2,1)
                int3c_ip2 = tmp = None
            for j0, (q0, q1) in enumerate(auxslices[:,2:]):
                _ej  = 0
                # (00|0)(1|1)(0|00)
                # (00|0)(1|0)(0|1)(0|00)
                _ej += .5 * np.einsum('p,xypq,q->xy', rhoj0_P[p0:p1],
                                      int2c_ip_ip[:,:,p0:p1,q0:q1], rhoj0_P[q0:q1])
                # (00|1)(1|0)(0|00)
                _ej -= lib.einsum('xp,yp->xy', rhoj1[:,q0:q1], wj0_01[:,q0:q1])
                # (00|1)(1|00)
                _ej += .5 * lib.einsum('xp,py->xy', rhoj1[:,q0:q1], wj_ip2[q0:q1])
                # (00|0)(0|1)(1|0)(0|00)
                _ej += .5 * np.einsum('xp,yp->xy', rhoj0_01[:,q0:q1], wj0_01[:,q0:q1])
                # (00|1)(0|1)(0|00)
                _ej -= lib.einsum('yqp,q,xp->xy', int2c_ip1[:,q0:q1], rhoj0_P[q0:q1], rhoj1)
                # (00|0)(1|0)(1|0)(0|00)
                _ej += np.einsum('xp,yp->xy', rhoj0_10[:,q0:q1], wj0_01[:,q0:q1])
                ej[i0,j0] += _ej
                ej[j0,i0] += _ej.T
                if with_k:
                    # (00|0)(1|1)(0|00)
                    # (00|0)(1|0)(0|1)(0|00)
                    _ek  = .5 * np.einsum('pq,xypq->xy', rho2c_0[p0:p1,q0:q1],
                                          int2c_ip_ip[:,:,p0:p1,q0:q1])
                    # (00|1)(0|1)(0|00)
                    # (00|1)(1|0)(0|00)
                    # (00|0)(0|1)(1|0)(0|00)
                    # (00|0)(1|0)(1|0)(0|00)
                    _ek += np.einsum('xpq,ypq->xy', rho2c_1[:,q0:q1], int2c_ip1[:,q0:q1])
                    # (00|1)(1|00)
                    _ek += .5 * lib.einsum('pxij,pq,qyij->xy',
                                           wk_ip2_P__[p0:p1], int2c_inv[p0:p1,q0:q1],
                                           wk_ip2_P__[q0:q1])
                    ek[i0,j0] += _ek * .5
                    ek[j0,i0] += _ek.T * .5

    for i0, ia in enumerate(atmlst):
        for j0 in range(i0):
            e1[j0,i0] = e1[i0,j0].T
            ej[j0,i0] = ej[i0,j0].T
            ek[j0,i0] = ek[i0,j0].T

    log.timer('RHF partial hessian', *time0)
    return e1, ej, ek


def make_h1(hessobj, mo_coeff, mo_occ, chkfile=None, atmlst=None, verbose=None):
    mol = hessobj.mol
    h1ao = [None] * mol.natm
    for ia, h1, vj1, vk1 in _gen_jk(hessobj, mo_coeff, mo_occ, chkfile,
                                    atmlst, verbose, True):
        h1 += vj1 - vk1 * .5
        h1ao[ia] = h1
    return h1ao

def _gen_jk(hessobj, mo_coeff, mo_occ, chkfile=None, atmlst=None,
            verbose=None, with_k=True):
    mol = hessobj.mol
    if atmlst is None:
        atmlst = range(mol.natm)

    with_df = hessobj.base.with_df
    auxmol = with_df.auxmol
    if auxmol is None:
        auxmol = df.addons.make_auxmol(with_df.mol, with_df.auxbasis)
    nbas = mol.nbas
    auxslices = auxmol.aoslice_by_atom()
    aux_loc = auxmol.ao_loc

    nao, nmo = mo_coeff.shape
    mocc = mo_coeff[:,mo_occ>0]
    nocc = mocc.shape[1]
    mocc_2 = np.einsum('pi,i->pi', mocc, mo_occ[mo_occ>0]**.5)
    dm0 = numpy.dot(mocc, mocc.T) * 2
    hcore_deriv = hessobj.base.nuc_grad_method().hcore_generator(mol)
    get_int3c = _int3c_wrapper(mol, auxmol, 'int3c2e', 's1')
    aoslices = mol.aoslice_by_atom()
    naux = auxmol.nao

    ftmp = lib.H5TmpFile()
    rho0_Pij = ftmp.create_group('rho0_Pij')
    wj_ip1_pij = ftmp.create_group('wj_ip1_pij')
    int2c = auxmol.intor('int2c2e', aosym='s1')
    solve_j2c = _gen_metric_solver(int2c)
    int2c = None
    int2c_ip1 = auxmol.intor('int2c2e_ip1', aosym='s1')
    rhoj0_P = 0
    if with_k:
        rhok0_Pl_ = np.empty((naux,nao,nocc))
    for i, (shl0, shl1, p0, p1) in enumerate(aoslices):
        int3c = get_int3c((shl0, shl1, 0, nbas, 0, auxmol.nbas))
        coef3c = solve_j2c(int3c.reshape(-1,naux).T)
        rho0_Pij['%.4d'%i] = coef3c = coef3c.reshape(naux,p1-p0,nao)
        rhoj0_P += np.einsum('pkl,kl->p', coef3c, dm0[p0:p1])
        if with_k:
            rhok0_Pl_[:,p0:p1] = lib.einsum('pij,jk->pik', coef3c, mocc_2)
        if hessobj.auxbasis_response:
            wj_ip1_pij['%.4d'%i] = lib.einsum('xqp,pij->qixj', int2c_ip1, coef3c)
    int3c = coef3c = None

    get_int3c_ip1 = _int3c_wrapper(mol, auxmol, 'int3c2e_ip1', 's1')
    get_int3c_ip2 = _int3c_wrapper(mol, auxmol, 'int3c2e_ip2', 's1')
    aux_ranges = ao2mo.outcore.balance_partition(auxmol.ao_loc, 480)
    vk1_buf = np.zeros((3,nao,nao))
    vj1_buf = np.zeros((mol.natm,3,nao,nao))
    for shl0, shl1, nL in aux_ranges:
        shls_slice = (0, nbas, 0, nbas, shl0, shl1)
        p0, p1 = aux_loc[shl0], aux_loc[shl1]
        int3c_ip1 = get_int3c_ip1(shls_slice)
        coef3c = _load_dim0(rho0_Pij, p0, p1)
        for i, (shl0, shl1, q0, q1) in enumerate(aoslices):
            wj1 = np.einsum('xijp,ji->xp', int3c_ip1[:,q0:q1], dm0[:,q0:q1])
            vj1_buf[i] += np.einsum('xp,pij->xij', wj1, coef3c)
        if with_k:
            rhok0_PlJ = lib.einsum('plj,Jj->plJ', rhok0_Pl_[p0:p1], mocc_2)
            vk1_buf += lib.einsum('xijp,plj->xil', int3c_ip1, rhok0_PlJ)
        int3c_ip1 = None
    vj1_buf = ftmp['vj1_buf'] = vj1_buf

    vk1 = np.zeros((3,nao,nao))
    for i0, ia in enumerate(atmlst):
        shl0, shl1, p0, p1 = aoslices[ia]
        shls_slice = (shl0, shl1, 0, nbas, 0, auxmol.nbas)
        int3c_ip1 = get_int3c_ip1(shls_slice)
        vj1 = -np.asarray(vj1_buf[ia])
        vj1[:,p0:p1] -= np.einsum('xijp,p->xij', int3c_ip1, rhoj0_P)
        if with_k:
            rhok0_PlJ = lib.einsum('plj,Jj->plJ', rhok0_Pl_, mocc_2[p0:p1])
            vk1 = -lib.einsum('xijp,pki->xkj', int3c_ip1, rhok0_PlJ)
            vk1[:,p0:p1] -= vk1_buf[:,p0:p1]

        if hessobj.auxbasis_response:
            shl0, shl1, q0, q1 = auxslices[ia]
            shls_slice = (0, nbas, 0, nbas, shl0, shl1)
            int3c_ip2 = get_int3c_ip2(shls_slice)
            rhoj1 = np.einsum('xijp,ji->xp', int3c_ip2, dm0)
            coef3c = _load_dim0(rho0_Pij, q0, q1)
            pij = _load_dim0(wj_ip1_pij, q0, q1)
            vj1 += .5 * np.einsum('pij,xp->xij', coef3c, -rhoj1)
            vj1 += .5 * np.einsum('xijp,p->xij', int3c_ip2, -rhoj0_P[q0:q1])
            vj1 -= .5 * lib.einsum('xpq,q,pij->xij', int2c_ip1[:,q0:q1], -rhoj0_P, coef3c)
            vj1 -= .5 * lib.einsum('pixj,p->xij', pij, -rhoj0_P[q0:q1])
            if with_k:
                rhok0_PlJ = lib.einsum('plj,Jj->plJ', rhok0_Pl_[q0:q1], mocc_2)
                vk1 -= lib.einsum('plj,xijp->xil', rhok0_PlJ, int3c_ip2)
                vk1 += lib.einsum('pjxi,plj->xil', pij, rhok0_PlJ)
        rhok0_PlJ = pij = coef3c = int3c_ip1 = None

        vj1 = vj1 + vj1.transpose(0,2,1)
        if with_k:
            vk1 = vk1 + vk1.transpose(0,2,1)
        h1 = hcore_deriv(ia)
        yield ia, h1, vj1, vk1

def _load_dim0(dat, p0, p1):
    return np.hstack([dat[x][p0:p1] for x in dat])


class Hessian(rhf_hess.Hessian):
    '''Non-relativistic restricted Hartree-Fock hessian'''
    def __init__(self, mf):
        rhf_hess.Hessian.__init__(self, mf)

    auxbasis_response = 1
    partial_hess_elec = partial_hess_elec
    make_h1 = make_h1

#TODO: Insert into DF class


if __name__ == '__main__':
    from pyscf import gto
    from pyscf import scf

    mol = gto.Mole()
    mol.verbose = 0
    mol.output = None
    mol.atom = [
        [1 , (1. ,  0.     , 0.000)],
        [1 , (0. ,  1.     , 0.000)],
        [1 , (0. , -1.517  , 1.177)],
        [1 , (0. ,  1.517  , 1.177)] ]
    mol.basis = '631g'
    mol.unit = 'B'
    mol.build()
    mf = scf.RHF(mol).density_fit()
    mf.conv_tol = 1e-14
    mf.scf()
    n3 = mol.natm * 3
    hobj = Hessian(mf)
    e2 = hobj.kernel()
    ref = scf.RHF(mol).run().Hessian().kernel()
    print(abs(e2-ref).max())
    print(lib.finger(e2) - 0.7232739558365785)
    e2 = hobj.set(auxbasis_response=2).kernel()
    print(abs(e2-ref).max())
    print(lib.finger(e2) - 0.72321237584876141)
