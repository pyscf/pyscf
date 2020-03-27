#!/usr/bin/env python
# Copyright 2014-2020 The PySCF Developers. All Rights Reserved.
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
#         Jun Yang
#

import numpy
from pyscf import lib

#einsum = numpy.einsum
einsum = lib.einsum

def _gamma1_intermediates(mycc, t1, t2, l1, l2):
    doo  =-einsum('ie,je->ij', l1, t1)
    doo -= einsum('imef,jmef->ij', l2, t2) * .5

    dvv  = einsum('ma,mb->ab', t1, l1)
    dvv += einsum('mnea,mneb->ab', t2, l2) * .5

    xt1  = einsum('mnef,inef->mi', l2, t2) * .5
    xt2  = einsum('mnfa,mnfe->ae', t2, l2) * .5
    xt2 += einsum('ma,me->ae', t1, l1)
    dvo  = einsum('imae,me->ai', t2, l1)
    dvo -= einsum('mi,ma->ai', xt1, t1)
    dvo -= einsum('ie,ae->ai', t1, xt2)
    dvo += t1.T

    dov = l1

    return doo, dov, dvo, dvv

# gamma2 intermediates in Chemist's notation
# When computing intermediates, the convention
# dm2[q,p,s,r] = <p^\dagger r^\dagger s q> is assumed in this function.
# It changes to dm2[p,q,r,s] = <p^\dagger r^\dagger s q> in _make_rdm2
def _gamma2_intermediates(mycc, t1, t2, l1, l2):
    tau = t2 + einsum('ia,jb->ijab', t1, t1) * 2
    miajb = einsum('ikac,kjcb->iajb', l2, t2)

    goovv = 0.25 * (l2.conj() + tau)
    tmp = einsum('kc,kica->ia', l1, t2)
    goovv += einsum('ia,jb->ijab', tmp, t1)
    tmp = einsum('kc,kb->cb', l1, t1)
    goovv += einsum('cb,ijca->ijab', tmp, t2) * .5
    tmp = einsum('kc,jc->kj', l1, t1)
    goovv += einsum('kiab,kj->ijab', tau, tmp) * .5
    tmp = numpy.einsum('ldjd->lj', miajb)
    goovv -= einsum('lj,liba->ijab', tmp, tau) * .25
    tmp = numpy.einsum('ldlb->db', miajb)
    goovv -= einsum('db,jida->ijab', tmp, tau) * .25
    goovv -= einsum('ldia,ljbd->ijab', miajb, tau) * .5
    tmp = einsum('klcd,ijcd->ijkl', l2, tau) * .25**2
    goovv += einsum('ijkl,klab->ijab', tmp, tau)
    goovv = goovv.conj()

    gvvvv = einsum('ijab,ijcd->abcd', tau, l2) * 0.125
    goooo = einsum('klab,ijab->klij', l2, tau) * 0.125

    gooov  = einsum('jkba,ib->jkia', tau, l1) * -0.25
    gooov += einsum('iljk,la->jkia', goooo, t1)
    tmp = numpy.einsum('icjc->ij', miajb) * .25
    gooov -= einsum('ij,ka->jkia', tmp, t1)
    gooov += einsum('icja,kc->jkia', miajb, t1) * .5
    gooov = gooov.conj()
    gooov += einsum('jkab,ib->jkia', l2, t1) * .25

    govvo  = einsum('ia,jb->ibaj', l1, t1)
    govvo += numpy.einsum('iajb->ibaj', miajb)
    govvo -= einsum('ikac,jc,kb->ibaj', l2, t1, t1)

    govvv  = einsum('ja,ijcb->iacb', l1, tau) * .25
    govvv += einsum('bcad,id->iabc', gvvvv, t1)
    tmp = numpy.einsum('kakb->ab', miajb) * .25
    govvv += einsum('ab,ic->iacb', tmp, t1)
    govvv += einsum('kaib,kc->iabc', miajb, t1) * .5
    govvv = govvv.conj()
    govvv += einsum('ijbc,ja->iabc', l2, t1) * .25

    dovov = goovv.transpose(0,2,1,3) - goovv.transpose(0,3,1,2)
    dvvvv = gvvvv.transpose(0,2,1,3) - gvvvv.transpose(0,3,1,2)
    doooo = goooo.transpose(0,2,1,3) - goooo.transpose(0,3,1,2)
    dovvv = govvv.transpose(0,2,1,3) - govvv.transpose(0,3,1,2)
    dooov = gooov.transpose(0,2,1,3) - gooov.transpose(1,2,0,3)
    dovvo = govvo.transpose(0,2,1,3)
    dovov =(dovov + dovov.transpose(2,3,0,1)) * .5
    dvvvv = dvvvv + dvvvv.transpose(1,0,3,2).conj()
    doooo = doooo + doooo.transpose(1,0,3,2).conj()
    dovvo =(dovvo + dovvo.transpose(3,2,1,0).conj()) * .5
    doovv = None # = -dovvo.transpose(0,3,2,1)
    dvvov = None
    return (dovov, dvvvv, doooo, doovv, dovvo, dvvov, dovvv, dooov)

def make_rdm1(mycc, t1, t2, l1, l2, ao_repr=False):
    r'''
    One-particle density matrix in the molecular spin-orbital representation
    (the occupied-virtual blocks from the orbital response contribution are
    not included).

    dm1[p,q] = <q^\dagger p>  (p,q are spin-orbitals)

    The convention of 1-pdm is based on McWeeney's book, Eq (5.4.20).
    The contraction between 1-particle Hamiltonian and rdm1 is
    E = einsum('pq,qp', h1, rdm1)
    '''
    d1 = _gamma1_intermediates(mycc, t1, t2, l1, l2)
    return _make_rdm1(mycc, d1, with_frozen=True, ao_repr=ao_repr)

def make_rdm2(mycc, t1, t2, l1, l2, ao_repr=False):
    r'''
    Two-particle density matrix in the molecular spin-orbital representation

    dm2[p,q,r,s] = <p^\dagger r^\dagger s q>

    where p,q,r,s are spin-orbitals. p,q correspond to one particle and r,s
    correspond to another particle.  The contraction between ERIs (in
    Chemist's notation) and rdm2 is
    E = einsum('pqrs,pqrs', eri, rdm2)
    '''
    d1 = _gamma1_intermediates(mycc, t1, t2, l1, l2)
    d2 = _gamma2_intermediates(mycc, t1, t2, l1, l2)
    return _make_rdm2(mycc, d1, d2, with_dm1=True, with_frozen=True,
                      ao_repr=ao_repr)

def _make_rdm1(mycc, d1, with_frozen=True, ao_repr=False):
    r'''
    One-particle density matrix in the molecular spin-orbital representation
    (the occupied-virtual blocks from the orbital response contribution are
    not included).

    dm1[p,q] = <q^\dagger p>  (p,q are spin-orbitals)

    The convention of 1-pdm is based on McWeeney's book, Eq (5.4.20).
    The contraction between 1-particle Hamiltonian and rdm1 is
    E = einsum('pq,qp', h1, rdm1)
    '''
    doo, dov, dvo, dvv = d1
    nocc, nvir = dov.shape
    nmo = nocc + nvir

    dm1 = numpy.empty((nmo,nmo), dtype=doo.dtype)
    dm1[:nocc,:nocc] = doo + doo.conj().T
    dm1[:nocc,nocc:] = dov + dvo.conj().T
    dm1[nocc:,:nocc] = dm1[:nocc,nocc:].conj().T
    dm1[nocc:,nocc:] = dvv + dvv.conj().T
    dm1 *= .5
    dm1[numpy.diag_indices(nocc)] += 1

    if with_frozen and mycc.frozen is not None:
        nmo = mycc.mo_occ.size
        nocc = numpy.count_nonzero(mycc.mo_occ > 0)
        rdm1 = numpy.zeros((nmo,nmo), dtype=dm1.dtype)
        rdm1[numpy.diag_indices(nocc)] = 1
        moidx = numpy.where(mycc.get_frozen_mask())[0]
        rdm1[moidx[:,None],moidx] = dm1
        dm1 = rdm1

    if ao_repr:
        mo = mycc.mo_coeff
        dm1 = lib.einsum('pi,ij,qj->pq', mo, dm1, mo.conj())
    return dm1

def _make_rdm2(mycc, d1, d2, with_dm1=True, with_frozen=True, ao_repr=False):
    r'''
    dm2[p,q,r,s] = <p^\dagger r^\dagger s q>

    Note the contraction between ERIs (in Chemist's notation) and rdm2 is
    E = einsum('pqrs,pqrs', eri, rdm2)
    '''
    dovov, dvvvv, doooo, doovv, dovvo, dvvov, dovvv, dooov = d2
    nocc, nvir = dovov.shape[:2]
    nmo = nocc + nvir

    dm2 = numpy.empty((nmo,nmo,nmo,nmo), dtype=doooo.dtype)

    dovov = numpy.asarray(dovov)
    dm2[:nocc,nocc:,:nocc,nocc:] = dovov
    dm2[nocc:,:nocc,nocc:,:nocc] = dm2[:nocc,nocc:,:nocc,nocc:].transpose(1,0,3,2).conj()
    dovov = None

    dovvo = numpy.asarray(dovvo)
    dm2[:nocc,:nocc,nocc:,nocc:] =-dovvo.transpose(0,3,2,1)
    dm2[nocc:,nocc:,:nocc,:nocc] =-dovvo.transpose(2,1,0,3)
    dm2[:nocc,nocc:,nocc:,:nocc] = dovvo
    dm2[nocc:,:nocc,:nocc,nocc:] = dovvo.transpose(1,0,3,2).conj()
    dovvo = None

    dm2[nocc:,nocc:,nocc:,nocc:] = dvvvv
    dm2[:nocc,:nocc,:nocc,:nocc] = doooo

    dovvv = numpy.asarray(dovvv)
    dm2[:nocc,nocc:,nocc:,nocc:] = dovvv
    dm2[nocc:,nocc:,:nocc,nocc:] = dovvv.transpose(2,3,0,1)
    dm2[nocc:,nocc:,nocc:,:nocc] = dovvv.transpose(3,2,1,0).conj()
    dm2[nocc:,:nocc,nocc:,nocc:] = dovvv.transpose(1,0,3,2).conj()
    dovvv = None

    dooov = numpy.asarray(dooov)
    dm2[:nocc,:nocc,:nocc,nocc:] = dooov
    dm2[:nocc,nocc:,:nocc,:nocc] = dooov.transpose(2,3,0,1)
    dm2[:nocc,:nocc,nocc:,:nocc] = dooov.transpose(1,0,3,2).conj()
    dm2[nocc:,:nocc,:nocc,:nocc] = dooov.transpose(3,2,1,0).conj()

    if with_frozen and mycc.frozen is not None:
        nmo, nmo0 = mycc.mo_occ.size, nmo
        nocc = numpy.count_nonzero(mycc.mo_occ > 0)
        rdm2 = numpy.zeros((nmo,nmo,nmo,nmo), dtype=dm2.dtype)
        moidx = numpy.where(mycc.get_frozen_mask())[0]
        idx = (moidx.reshape(-1,1) * nmo + moidx).ravel()
        lib.takebak_2d(rdm2.reshape(nmo**2,nmo**2),
                       dm2.reshape(nmo0**2,nmo0**2), idx, idx)
        dm2 = rdm2

    if with_dm1:
        dm1 = _make_rdm1(mycc, d1, with_frozen)
        dm1[numpy.diag_indices(nocc)] -= 1

        for i in range(nocc):
# Be careful with the convention of dm1 and the transpose of dm2 at the end
            dm2[i,i,:,:] += dm1
            dm2[:,:,i,i] += dm1
            dm2[:,i,i,:] -= dm1
            dm2[i,:,:,i] -= dm1.T

        for i in range(nocc):
            for j in range(nocc):
                dm2[i,i,j,j] += 1
                dm2[i,j,j,i] -= 1

    # dm2 was computed as dm2[p,q,r,s] = < p^\dagger r^\dagger s q > in the
    # above. Transposing it so that it be contracted with ERIs (in Chemist's
    # notation):
    #   E = einsum('pqrs,pqrs', eri, rdm2)
    dm2 = dm2.transpose(1,0,3,2)
    if ao_repr:
        from pyscf.cc import ccsd_rdm
        dm2 = ccsd_rdm._rdm2_mo2ao(dm2, mycc.mo_coeff)
    return dm2


if __name__ == '__main__':
    from functools import reduce
    from pyscf import gto
    from pyscf import scf
    from pyscf import ao2mo
    from pyscf.cc import gccsd

    mol = gto.Mole()
    mol.atom = [
        [8 , (0. , 0.     , 0.)],
        [1 , (0. , -0.757 , 0.587)],
        [1 , (0. , 0.757  , 0.587)]]
    mol.basis = '631g'
    mol.spin = 2
    mol.build()
    mf = scf.UHF(mol).run(conv_tol=1.)
    mf = scf.addons.convert_to_ghf(mf)

    mycc = gccsd.GCCSD(mf)
    ecc, t1, t2 = mycc.kernel()
    l1, l2 = mycc.solve_lambda()
    dm1 = make_rdm1(mycc, t1, t2, l1, l2)
    dm2 = make_rdm2(mycc, t1, t2, l1, l2)
    nao = mol.nao_nr()
    mo_a = mf.mo_coeff[:nao]
    mo_b = mf.mo_coeff[nao:]
    nmo = mo_a.shape[1]
    eri = ao2mo.kernel(mf._eri, mo_a+mo_b, compact=False).reshape([nmo]*4)
    orbspin = mf.mo_coeff.orbspin
    sym_forbid = (orbspin[:,None] != orbspin)
    eri[sym_forbid,:,:] = 0
    eri[:,:,sym_forbid] = 0
    hcore = scf.RHF(mol).get_hcore()
    h1 = reduce(numpy.dot, (mo_a.T.conj(), hcore, mo_a))
    h1+= reduce(numpy.dot, (mo_b.T.conj(), hcore, mo_b))
    e1 = numpy.einsum('ij,ji', h1, dm1)
    e1+= numpy.einsum('ijkl,ijkl', eri, dm2) * .5
    e1+= mol.energy_nuc()
    print(e1 - mycc.e_tot)

    #TODO: test 1pdm, 2pdm against FCI
