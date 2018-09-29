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

'''
MP2-F12 (In testing)

Refs:
* JCC 32  2492
* JCP 139 084112

With strong orthogonalization ansatz 2
'''

import time
from functools import reduce
import numpy
import scipy.linalg
from pyscf import lib
from pyscf.lib import logger
from pyscf import gto
from pyscf import ao2mo
from pyscf.scf import jk
from pyscf.mp import mp2

# The cabs space, the complimentary space to the OBS.
def find_cabs(mol, auxmol, lindep=1e-8):
    cabs_mol = gto.conc_mol(mol, auxmol)
    nao = mol.nao_nr()
    s = cabs_mol.intor_symmetric('int1e_ovlp')

    ls12 = scipy.linalg.solve(s[:nao,:nao], s[:nao,nao:], sym_pos=True)
    s[nao:,nao:] -= s[nao:,:nao].dot(ls12)
    w, v = scipy.linalg.eigh(s[nao:,nao:])
    c2 = v[:,w>lindep]/numpy.sqrt(w[w>lindep])
    c1 = ls12.dot(c2)
    return cabs_mol, numpy.vstack((-c1,c2))

def trans(eri, mos):
    naoi, nmoi = mos[0].shape
    naoj, nmoj = mos[1].shape
    naok, nmok = mos[2].shape
    naol, nmol = mos[3].shape
    eri1 = numpy.dot(mos[0].T, eri.reshape(naoi,-1))
    eri1 = eri1.reshape(nmoi,naoj,naok,naol)

    eri1 = numpy.dot(mos[1].T, eri1.transpose(1,0,2,3).reshape(naoj,-1))
    eri1 = eri1.reshape(nmoj,nmoi,naok,naol).transpose(1,0,2,3)

    eri1 = numpy.dot(eri1.transpose(0,1,3,2).reshape(-1,naok), mos[2])
    eri1 = eri1.reshape(nmoi,nmoj,naol,nmok).transpose(0,1,3,2)

    eri1 = numpy.dot(eri1.reshape(-1,naol), mos[3])
    eri1 = eri1.reshape(nmoi,nmoj,nmok,nmol)
    return eri1

def energy_f12(mf, auxmol, zeta):
    logger.info(mf, '******** MP2-F12 (In testing) ********')
    mol = mf.mol
    mo_coeff = mf.mo_coeff
    mo_energy = mf.mo_energy
    nocc = numpy.count_nonzero(mf.mo_occ == 2)

    cabs_mol, cabs_coeff = find_cabs(mol, auxmol)
    nao, nmo = mo_coeff.shape
    nca = cabs_coeff.shape[0]
    mo_o = mo_coeff[:,:nocc]
    mo_v = mo_coeff[:,nocc:]
    Pcoeff = numpy.vstack((mo_coeff, numpy.zeros((nca-nao, nmo))))
    Pcoeff = numpy.hstack((Pcoeff, cabs_coeff))
    obs = (0, mol.nbas)
    cbs = (0, cabs_mol.nbas)

    mol.set_f12_zeta(zeta)
    Y = mol.intor('int2e_yp')
    Y = trans(Y, [mo_o]*4)

    cabs_mol.set_f12_zeta(zeta)
    R = cabs_mol.intor('int2e_stg', shls_slice=obs+cbs+obs+cbs)
    RmPnQ = trans(R, [mo_o, Pcoeff, mo_o, Pcoeff])
    Rmpnq = RmPnQ[:,:nmo,:,:nmo]
    Rmlnc = RmPnQ[:nocc,:nocc,:nocc,nmo:]
    Rmcnl = Rmlnc.transpose(2,3,0,1)
    Rpiqj = Rmpnq.transpose(1,0,3,2)
    Rlicj = Rmlnc.transpose(0,1,3,2)
    Rcilj = Rlicj.transpose(2,3,0,1)
    RRiQj = RmPnQ.transpose(1,0,3,2)
    RmPnk = RmPnQ[:,:,:,:nocc]
    RQikj = RmPnk.transpose(1,0,2,3)
    Rmknc = Rmlnc
    Rmpna = Rmpnq[:,:,:,nocc:nmo]
    Rqiaj = Rpiqj[:,:,nocc:nmo,:]
    RPicj = RRiQj[:,:,nmo:,:]
    Rmcnb = RmPnQ[:,nmo:,:,nocc:nmo]
    Rpibj = Rqiaj

    cabs_mol.set_f12_zeta(zeta*2)
    Rbar = cabs_mol.intor('int2e_stg', shls_slice=cbs+obs+obs+obs)
    Rbar = Rbar.reshape(nca,nao,nao,nao)
    Rbar_minj = trans(Rbar[:nao], [mo_o]*4)
    Rbar_miPj = trans(Rbar, [Pcoeff, mo_o, mo_o, mo_o]).transpose(2,3,0,1)
    tau = Rbar[:nao] * zeta**2
    tau = trans(tau, [mo_o]*4)

    v = cabs_mol.intor('int2e', shls_slice=cbs+obs+obs+obs)
    v = v.reshape(nca,nao,nao,nao)
    vpiqj = trans(v[:nao], [mo_coeff, mo_o, mo_coeff, mo_o])
    vlicj = trans(v, [cabs_coeff, mo_o, mo_o, mo_o]).transpose(2,3,0,1)
    vcilj = vlicj.transpose(2,3,0,1)

    fPQ = mf.get_hcore(cabs_mol)
    dm = numpy.dot(mo_o, mo_o.T) * 2
    v = cabs_mol.intor('int2e', shls_slice=cbs+cbs+obs+obs)
    v = v.reshape(nca,nca,nao,nao)
    fPQ += numpy.einsum('pqij,ji->pq', v, dm)
    fPQ = reduce(numpy.dot, (Pcoeff.T, fPQ, Pcoeff))
    v = cabs_mol.intor('int2e', shls_slice=cbs+obs+obs+cbs)
    v = v.reshape(nca,nao,nao,nca)
    kPQ = numpy.einsum('pijq,ij->pq', v, dm)*.5
    kPQ = reduce(numpy.dot, (Pcoeff.T, kPQ, Pcoeff))
    tPQ = cabs_mol.intor_symmetric('int1e_kin')
    tPQ = reduce(numpy.dot, (Pcoeff.T, tPQ, Pcoeff))
    hPQ = fPQ - tPQ  # hartree term
    fPQ = hPQ - kPQ

    tminj = numpy.zeros([nocc]*4)
    for i in range(nocc):
        for j in range(nocc):
            tminj[i,i,j,j] = -3./(8*zeta)
            tminj[i,j,j,i] = -1./(8*zeta)
        tminj[i,i,i,i] = -.5/zeta

    V = Y
    V-= numpy.einsum('mpnq,piqj->minj', Rmpnq, vpiqj)
    V-= numpy.einsum('mlnc,licj->minj', Rmlnc, vlicj)
    V-= numpy.einsum('mcnl,cilj->minj', Rmcnl, vcilj)
    emp2_f12 = numpy.einsum('minj,minj', V, tminj) * 4
    emp2_f12-= numpy.einsum('minj,nimj', V, tminj) * 2

    X = Rbar_minj
    X-= numpy.einsum('mpnq,piqj->minj', Rmpnq, Rpiqj)
    X-= numpy.einsum('mlnc,licj->minj', Rmlnc, Rlicj)
    X-= numpy.einsum('mcnl,cilj->minj', Rmcnl, Rcilj)

    tmp = numpy.einsum('miPj,nP->minj', Rbar_miPj, hPQ[:nocc])
    B   = (tmp + tmp.transpose(1,0,3,2)) * .5
    tmp = numpy.einsum('mPnQ,PR->mRnQ', RmPnQ, kPQ)
    B  -= numpy.einsum('mRnQ,RiQj->minj', tmp, RRiQj)
    tmp = numpy.einsum('mPnk,PQ->mQnk', RmPnk, fPQ)
    B  -= numpy.einsum('mQnk,Qikj->minj', tmp, RQikj)
    tmp = numpy.einsum('mknc,kl->mlnc', Rmknc, fPQ[:nocc,:nocc])
    B  += numpy.einsum('mlnc,licj->minj', tmp, Rlicj)
    tmp = numpy.einsum('mpna,pq->mqna', Rmpna, fPQ[:nmo,:nmo])
    B  -= numpy.einsum('mqna,qiaj->minj', tmp, Rqiaj)
    tmp = numpy.einsum('mknc,kP->mPnc', Rmknc, fPQ[:nocc])
    tmp1= numpy.einsum('mPnc,Picj->minj', tmp, RPicj)
    tmp = numpy.einsum('mcnb,cp->mpnb', Rmcnb, fPQ[nmo:,:nmo])
    tmp1+= numpy.einsum('mpnb,pibj->minj', tmp, Rpibj)
    B  -= tmp1 + tmp1.transpose(1,0,3,2)
    B   = B + B.transpose(2,3,0,1)
    B  += tau

    e_mn = lib.direct_sum('i+j->ij', mo_energy[:nocc], mo_energy[:nocc])
    tmp = numpy.einsum('mknl,kilj->minj', tminj, B)
    emp2_f12+= numpy.einsum('minj,minj', tmp, tminj) * 2
    emp2_f12-= numpy.einsum('minj,nimj', tmp, tminj)
    tmp = numpy.einsum('mknl,kilj->minj', tminj, X)
    emp2_f12-= numpy.einsum('mn,minj,minj', e_mn, tmp, tminj) * 2
    emp2_f12+= numpy.einsum('mn,minj,nimj', e_mn, tmp, tminj)
    return emp2_f12

if __name__ == '__main__':
    from pyscf import scf
    from pyscf import gto
    mol = gto.Mole()
    #mol.atom = [
    #    [8 , (0. , 0.     , 0.)],
    #    [1 , (0. , -0.757 , 0.587)],
    #    [1 , (0. , 0.757  , 0.587)]]
    mol.atom = 'Ne 0 0 0'

    mol.basis = 'cc-pvdz'
    mol.build()
    mf = scf.RHF(mol)
    print(mf.scf())

    e = mp2.MP2(mf).kernel()[0]
    auxmol = mol.copy()
    #auxmol.basis = 'cc-pVDZ-F12-OptRI'
    auxmol.basis = ('ccpvdz-fit', 'cc-pVDZ-F12-OptRI')
    #auxmol.basis = 'cc-pVTZ'
    auxmol.build(False, False)
    print('MP2', e)
    e+= energy_f12(mf, auxmol, 1.)
    print('MP2-F12', e)
    print('e_tot', e+mf.e_tot)
