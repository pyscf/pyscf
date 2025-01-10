#!/usr/bin/env python
# Copyright 2014-2019 The PySCF Developers. All Rights Reserved.
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
RCCSD

Ref: JCP 90, 1752 (1989); DOI:10.1063/1.456069
'''

from functools import reduce
import numpy
import scipy.linalg
from pyscf import lib
from pyscf import ao2mo
from pyscf.cc import ccsd_rdm
from pyscf.grad import ccsd as ccsd_grad
from pyscf.grad.mp2 import has_frozen_orbitals

def kernel(cc, t1, t2, l1, l2, eris=None):
    if eris is None:
        eris = _ERIS(cc, cc.mo_coeff)
    mol = cc.mol
    mo_coeff = cc.mo_coeff
    mo_energy = cc._scf.mo_energy
    nao, nmo = mo_coeff.shape
    nocc = numpy.count_nonzero(cc.mo_occ > 0)
    mo_e_o = mo_energy[:nocc]
    mo_e_v = mo_energy[nocc:]
    with_frozen = has_frozen_orbitals(cc)

    d1 = _gamma1_intermediates(cc, t1, t2, l1, l2)
    d2 = _gamma2_intermediates(cc, t1, t2, l1, l2)

    dm2 = ccsd_rdm._make_rdm2(cc, d1, d2, with_dm1=False, with_frozen=False)
    eri = ao2mo.restore(1, ao2mo.full(cc.mol, mo_coeff), nmo)
    Imat = numpy.einsum('jqrs,iqrs->ij', dm2, eri) * -1
    Ioo = Imat[:nocc,:nocc]
    Ivv = Imat[nocc:,nocc:]
    doo, dov, dvo, dvv = d1
    if with_frozen:
        OA, VA, OF, VF = index_frozen_active(cc)
        doo[OF[:,None],OA] = Ioo[OF[:,None],OA] / lib.direct_sum('i-j->ij', mo_e_o[OF], mo_e_o[OA])
        doo[OA[:,None],OF] = Ioo[OA[:,None],OF] / lib.direct_sum('i-j->ij', mo_e_o[OA], mo_e_o[OF])
        dvv[VF[:,None],VA] = Ivv[VF[:,None],VA] / lib.direct_sum('a-b->ab', mo_e_v[VF], mo_e_v[VA])
        dvv[VA[:,None],VF] = Ivv[VA[:,None],VF] / lib.direct_sum('a-b->ab', mo_e_v[VA], mo_e_v[VF])
    dm1 = scipy.linalg.block_diag(doo+doo.T, dvv+dvv.T)
    dm1ao = reduce(numpy.dot, (mo_coeff, dm1, mo_coeff.T))
    vj, vk = cc._scf.get_jk(cc.mol, dm1ao)
    Xvo = reduce(numpy.dot, (mo_coeff[:,nocc:].T, vj*2-vk, mo_coeff[:,:nocc]))
    Xvo += Imat[:nocc,nocc:].T - Imat[nocc:,:nocc]

    dm1 += ccsd_grad._response_dm1(cc, Xvo, eris)
    Imat[nocc:,:nocc] = Imat[:nocc,nocc:].T

    h1 =-(mol.intor('int1e_ipkin', comp=3) +
          mol.intor('int1e_ipnuc', comp=3))
    s1 =-mol.intor('int1e_ipovlp', comp=3)
    #zeta = lib.direct_sum('i-j->ij', mo_energy, mo_energy)
    eri1 = mol.intor('int2e_ip1', comp=3).reshape(3,nao,nao,nao,nao)
    eri1 = numpy.einsum('xipkl,pj->xijkl', eri1, mo_coeff)
    eri1 = numpy.einsum('xijpl,pk->xijkl', eri1, mo_coeff)
    eri1 = numpy.einsum('xijkp,pl->xijkl', eri1, mo_coeff)
    g0 = ao2mo.restore(1, ao2mo.full(mol, mo_coeff), nmo)

    de = numpy.empty((mol.natm,3))
    for k,(sh0, sh1, p0, p1) in enumerate(mol.offset_nr_by_atom()):
        mol.set_rinv_origin(mol.atom_coord(k))
        vrinv = -mol.atom_charge(k) * mol.intor('int1e_iprinv', comp=3)

# 2e AO integrals dot 2pdm
        de2 = numpy.zeros(3)
        for i in range(3):
            g1 = numpy.einsum('pjkl,pi->ijkl', eri1[i,p0:p1], mo_coeff[p0:p1])
            g1 = g1 + g1.transpose(1,0,2,3)
            g1 = g1 + g1.transpose(2,3,0,1)
            g1 *= -1
            hx =(numpy.einsum('pq,pi,qj->ij', h1[i,p0:p1], mo_coeff[p0:p1], mo_coeff) +
                 reduce(numpy.dot, (mo_coeff.T, vrinv[i], mo_coeff)))
            hx = hx + hx.T
            sx = numpy.einsum('pq,pi,qj->ij', s1[i,p0:p1], mo_coeff[p0:p1], mo_coeff)
            sx = sx + sx.T

            fij =(hx[:nocc,:nocc]
                  - numpy.einsum('ij,j->ij', sx[:nocc,:nocc], mo_e_o) * .5
                  - numpy.einsum('ij,i->ij', sx[:nocc,:nocc], mo_e_o) * .5
                  - numpy.einsum('kl,ijlk->ij', sx[:nocc,:nocc],
                                 g0[:nocc,:nocc,:nocc,:nocc]) * 2
                  + numpy.einsum('kl,iklj->ij', sx[:nocc,:nocc],
                                 g0[:nocc,:nocc,:nocc,:nocc])
                  + numpy.einsum('ijkk->ij', g1[:nocc,:nocc,:nocc,:nocc]) * 2
                  - numpy.einsum('ikkj->ij', g1[:nocc,:nocc,:nocc,:nocc]))

            fab =(hx[nocc:,nocc:]
                  - numpy.einsum('ij,j->ij', sx[nocc:,nocc:], mo_e_v) * .5
                  - numpy.einsum('ij,i->ij', sx[nocc:,nocc:], mo_e_v) * .5
                  - numpy.einsum('kl,ijlk->ij', sx[:nocc,:nocc],
                                 g0[nocc:,nocc:,:nocc,:nocc]) * 2
                  + numpy.einsum('kl,iklj->ij', sx[:nocc,:nocc],
                                 g0[nocc:,:nocc,:nocc,nocc:])
                  + numpy.einsum('ijkk->ij', g1[nocc:,nocc:,:nocc,:nocc]) * 2
                  - numpy.einsum('ikkj->ij', g1[nocc:,:nocc,:nocc,nocc:]))

            if with_frozen:
                fij[OA[:,None],OF] -= numpy.einsum('ij,j->ij', sx[OA[:,None],OF], mo_e_o[OF]) * .5
                fij[OA[:,None],OF] += numpy.einsum('ij,i->ij', sx[OA[:,None],OF], mo_e_o[OA]) * .5
                fij[OF[:,None],OA] -= numpy.einsum('ij,j->ij', sx[OF[:,None],OA], mo_e_o[OA]) * .5
                fij[OF[:,None],OA] += numpy.einsum('ij,i->ij', sx[OF[:,None],OA], mo_e_o[OF]) * .5
                fab[VA[:,None],VF] -= numpy.einsum('ij,j->ij', sx[VA[:,None],VF], mo_e_v[VF]) * .5
                fab[VA[:,None],VF] += numpy.einsum('ij,i->ij', sx[VA[:,None],VF], mo_e_v[VA]) * .5
                fab[VF[:,None],VA] -= numpy.einsum('ij,j->ij', sx[VF[:,None],VA], mo_e_v[VA]) * .5
                fab[VF[:,None],VA] += numpy.einsum('ij,i->ij', sx[VF[:,None],VA], mo_e_v[VF]) * .5

            fai =(hx[nocc:,:nocc]
                  - numpy.einsum('ai,i->ai', sx[nocc:,:nocc], mo_e_o)
                  - numpy.einsum('kl,ijlk->ij', sx[:nocc,:nocc],
                                 g0[nocc:,:nocc,:nocc,:nocc]) * 2
                  + numpy.einsum('kl,iklj->ij', sx[:nocc,:nocc],
                                 g0[nocc:,:nocc,:nocc,:nocc])
                  + numpy.einsum('ijkk->ij', g1[nocc:,:nocc,:nocc,:nocc]) * 2
                  - numpy.einsum('ikkj->ij', g1[nocc:,:nocc,:nocc,:nocc]))

            f1 = numpy.zeros((nmo,nmo))
            f1[:nocc,:nocc] = fij
            f1[nocc:,nocc:] = fab
            f1[nocc:,:nocc] = fai
            f1[:nocc,nocc:] = fai.T
            de2[i] += numpy.einsum('ij,ji', f1, dm1)
            de2[i] += numpy.einsum('ij,ji', sx, Imat)
            de2[i] += numpy.einsum('iajb,iajb', dm2, g1) * .5

        de[k] = de2

    return de


class _ERIS:
    def __init__(self, cc, mo_coeff):
        nocc = numpy.count_nonzero(cc.mo_occ > 0)
        eri0 = ao2mo.full(cc._scf._eri, mo_coeff)
        eri0 = ao2mo.restore(1, eri0, mo_coeff.shape[1])
        eri0 = eri0.reshape((mo_coeff.shape[1],)*4)
        self.oooo = eri0[:nocc,:nocc,:nocc,:nocc].copy()
        self.ooov = eri0[:nocc,:nocc,:nocc,nocc:].copy()
        self.ovoo = eri0[:nocc,nocc:,:nocc,:nocc].copy()
        self.oovo = eri0[:nocc,:nocc,nocc:,:nocc].copy()
        self.oovv = eri0[:nocc,:nocc,nocc:,nocc:].copy()
        self.ovov = eri0[:nocc,nocc:,:nocc,nocc:].copy()
        self.ovvo = eri0[:nocc,nocc:,nocc:,:nocc].copy()
        self.ovvv = eri0[:nocc,nocc:,nocc:,nocc:].copy()
        self.vvvv = eri0[nocc:,nocc:,nocc:,nocc:].copy()
        self.vvvo = eri0[nocc:,nocc:,nocc:,:nocc].copy()
        self.vovv = eri0[nocc:,:nocc,nocc:,nocc:].copy()
        self.vvov = eri0[nocc:,nocc:,:nocc,nocc:].copy()
        self.vvoo = eri0[nocc:,nocc:,:nocc,:nocc].copy()
        self.voov = eri0[nocc:,:nocc,:nocc,nocc:].copy()
        self.vooo = eri0[nocc:,:nocc,:nocc,:nocc].copy()
        self.mo_coeff = mo_coeff
        self.fock = numpy.diag(cc._scf.mo_energy)

def index_frozen_active(cc):
    nocc = numpy.count_nonzero(cc.mo_occ > 0)
    moidx = cc.get_frozen_mask()
    OA = numpy.where( moidx[:nocc])[0] # occupied active orbitals
    OF = numpy.where(~moidx[:nocc])[0] # occupied frozen orbitals
    VA = numpy.where( moidx[nocc:])[0] # virtual active orbitals
    VF = numpy.where(~moidx[nocc:])[0] # virtual frozen orbitals
    return OA, VA, OF, VF

def _gamma1_intermediates(cc, t1, t2, l1, l2):
    d1 = ccsd_rdm._gamma1_intermediates(cc, t1, t2, l1, l2)
    if cc.frozen is None:
        return d1
    nocc = numpy.count_nonzero(cc.mo_occ>0)
    nvir = cc.mo_occ.size - nocc
    OA, VA, OF, VF = index_frozen_active(cc)
    doo = numpy.zeros((nocc,nocc))
    dov = numpy.zeros((nocc,nvir))
    dvo = numpy.zeros((nvir,nocc))
    dvv = numpy.zeros((nvir,nvir))
    doo[OA[:,None],OA] = d1[0]
    dov[OA[:,None],VA] = d1[1]
    dvo[VA[:,None],OA] = d1[2]
    dvv[VA[:,None],VA] = d1[3]
    return doo, dov, dvo, dvv

def _gamma2_intermediates(cc, t1, t2, l1, l2):
    d2 = ccsd_rdm._gamma2_intermediates(cc, t1, t2, l1, l2)
    nocc, nvir = t1.shape
    if cc.frozen is None:
        dovov, dvvvv, doooo, doovv, dovvo, dvvov, dovvv, dooov = d2
        dvvov = dovvv.transpose(2,3,0,1)
        dvvvv = ao2mo.restore(1, d2[1], nvir)
        return dovov, dvvvv, doooo, doovv, dovvo, dvvov, dovvv, dooov
    nocc0 = numpy.count_nonzero(cc.mo_occ>0)
    nvir0 = cc.mo_occ.size - nocc0
    OA, VA, OF, VF = index_frozen_active(cc)
    dovov = numpy.zeros((nocc0,nvir0,nocc0,nvir0))
    dvvvv = numpy.zeros((nvir0,nvir0,nvir0,nvir0))
    doooo = numpy.zeros((nocc0,nocc0,nocc0,nocc0))
    doovv = numpy.zeros((nocc0,nocc0,nvir0,nvir0))
    dovvo = numpy.zeros((nocc0,nvir0,nvir0,nocc0))
    dovvv = numpy.zeros((nocc0,nvir0,nvir0,nvir0))
    dooov = numpy.zeros((nocc0,nocc0,nocc0,nvir0))
    dovov[OA[:,None,None,None],VA[:,None,None],OA[:,None],VA] = d2[0]
    dvvvv[VA[:,None,None,None],VA[:,None,None],VA[:,None],VA] = ao2mo.restore(1, d2[1], nvir)
    doooo[OA[:,None,None,None],OA[:,None,None],OA[:,None],OA] = d2[2]
    doovv[OA[:,None,None,None],OA[:,None,None],VA[:,None],VA] = d2[3]
    dovvo[OA[:,None,None,None],VA[:,None,None],VA[:,None],OA] = d2[4]
    dovvv[OA[:,None,None,None],VA[:,None,None],VA[:,None],VA] = d2[6]
    dooov[OA[:,None,None,None],OA[:,None,None],OA[:,None],VA] = d2[7]
    dvvov = dovvv.transpose(2,3,0,1)
    return dovov, dvvvv, doooo, doovv, dovvo, dvvov, dovvv, dooov


if __name__ == '__main__':
    from pyscf import gto
    from pyscf import scf
    from pyscf.cc import ccsd
    from pyscf import grad

    mol = gto.M(
        verbose = 0,
        atom = [
            ["O" , (0. , 0.     , 0.)],
            [1   , (0. ,-0.757  , 0.587)],
            [1   , (0. , 0.757  , 0.587)]],
        basis = '631g'
    )
    mf = scf.RHF(mol).run()
    mycc = ccsd.CCSD(mf)
    ecc, t1, t2 = mycc.kernel()
    l1, l2 = mycc.solve_lambda()
    g1 = kernel(mycc, t1, t2, l1, l2)
    ghf = grad.RHF(mf).grad()
    print('gcc')
    print(ghf+g1)
    print(lib.fp(g1) - -0.042511000925747583)
#[[ 0   0                1.00950969e-02]
# [ 0   2.28063353e-02  -5.04754844e-03]
# [ 0  -2.28063353e-02  -5.04754844e-03]]

    print('-----------------------------------')
    mol = gto.M(
        verbose = 0,
        atom = [
            ["O" , (0. , 0.     , 0.)],
            [1   , (0. ,-0.757  , 0.587)],
            [1   , (0. , 0.757  , 0.587)]],
        basis = '631g'
    )
    mf = scf.RHF(mol).run()
    mycc = ccsd.CCSD(mf)
    mycc.frozen = [0,1,10,11,12]
    ecc, t1, t2 = mycc.kernel()
    l1, l2 = mycc.solve_lambda()
    g1 = kernel(mycc, t1, t2, l1, l2)
    ghf = grad.RHF(mf).grad()
    print('gcc')
    print(ghf+g1)
    print(lib.fp(g1) - 0.10048468674687236)
#[[ -7.81105940e-17   3.81840540e-15   1.20415540e-02]
# [  1.73095055e-16  -7.94568837e-02  -6.02077699e-03]
# [ -9.49844615e-17   7.94568837e-02  -6.02077699e-03]]

    r = 1.76
    mol = gto.M(
        verbose = 0,
        atom = '''H 0 0 0; H 0 0 %f''' % r,
        basis = '631g',
        unit = 'bohr')
    mf = scf.RHF(mol)
    mf.conv_tol = 1e-14
    ehf0 = mf.scf()
    ghf = grad.RHF(mf).grad()
    mycc = ccsd.CCSD(mf)
    ecc, t1, t2 = mycc.kernel()
    l1, l2 = mycc.solve_lambda()
    g1 = kernel(mycc, t1, t2, l1, l2)
    ghf = grad.RHF(mf).grad()
    print('ghf')
    print(ghf)
    print('gcc')
    print(g1) # 0.015643667024
    print('tot')
    print(ghf+g1) # -0.0708003526454

    mol = gto.M(
        verbose = 0,
        atom = '''H 0 0 0; H 0 0 %f''' % (r-.001),
        basis = '631g',
        unit = 'bohr')
    mf = scf.RHF(mol)
    mf.conv_tol = 1e-14
    ehf0 = mf.scf()
    mycc = ccsd.CCSD(mf)
    ecc0 = mycc.kernel()[0]

    mol = gto.M(
        verbose = 0,
        atom = '''H 0 0 0; H 0 0 %f''' % (r+.001),
        basis = '631g',
        unit = 'bohr')
    mf = scf.RHF(mol)
    mf.conv_tol = 1e-14
    ehf1 = mf.scf()
    mycc = ccsd.CCSD(mf)
    ecc1 = mycc.kernel()[0]
    print((ehf1-ehf0)*500 - ghf[1,2])
    print('decc', (ecc1-ecc0)*500 - g1[1,2])
    print('decc', (ehf1+ecc1-ehf0-ecc0)*500 - (ghf[1,2]+g1[1,2]))
