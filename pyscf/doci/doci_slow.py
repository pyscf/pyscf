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
Doubly occupied configuration interaction (DOCI)
'''

import numpy
from pyscf import lib
from pyscf import ao2mo
from pyscf.fci import cistring
from pyscf.fci import direct_spin1
from pyscf.fci import rdm


def contract_2e(eri, civec, norb, nelec, link_index=None):
    '''Compute E_{pq}E_{rs}|CI>'''
    if isinstance(nelec, (int, numpy.integer)):
        nelecb = nelec//2
        neleca = nelec - nelecb
    else:
        neleca, nelecb = nelec
    assert(neleca == nelecb)

    if link_index is None:
        link_index = cistring.gen_linkstr_index(range(norb), neleca)
    na = link_index.shape[0]
    t1 = numpy.zeros((norb,na))
    t2 = numpy.zeros((norb,norb,na))
    #:for str0, tab in enumerate(link_index):
    #:    for a, i, str1, sign in tab:
    #:        if a == i:
    #:            t1[i,str1] += civec[str0]
    #:        else:
    #:            t2[a,i,str1] += civec[str0]
    link1 = link_index[link_index[:,:,0] == link_index[:,:,1]].reshape(na,-1,4)
    link2 = link_index[link_index[:,:,0] != link_index[:,:,1]].reshape(na,-1,4)
    t1[link1[:,:,1],link1[:,:,2]] = civec[:,None]
    t2[link2[:,:,0],link2[:,:,1],link2[:,:,2]] = civec[:,None]

    eri = ao2mo.restore(1, eri, norb)
    # [(ia,ja|ib,jb) + (ib,jb|ia,ja)] ~ E_{ij}E_{ij} where i != j
    cinew = numpy.einsum('ijij,ijp->p', eri, t2) * 2

    # [(ia,ja|ja,ia) + (ib,jb|jb,ib)] ~ E_{ij}E_{ji} where i != j
    k_diag = numpy.einsum('ijji->ij', eri)
    t2 = numpy.einsum('ij,ijp->ijp', k_diag * 2, t2)

    # [(ia,ia|ja,ja) + (ia,ia|jb,jb) + ...] ~ E_{ii}E_{jj}
    j_diag = numpy.einsum('iijj->ij', eri)
    t1 = numpy.einsum('ij,jp->ip', j_diag, t1) * 4

    #:for str0, tab in enumerate(link_index):
    #:    for a, i, str1, sign in tab:
    #:        if a == i:
    #:            cinew[str0] += t1[i,str1]
    #:        else:
    #:            cinew[str0] += t2[a,i,str1]
    cinew += numpy.einsum('pi->p', t1[link1[:,:,1],link1[:,:,2]])
    cinew += numpy.einsum('pi->p', t2[link2[:,:,0],link2[:,:,1],link2[:,:,2]])
    return cinew

def make_hdiag(h1e, eri, norb, nelec, opt=None):
    if isinstance(nelec, (int, numpy.integer)):
        nelecb = nelec//2
        neleca = nelec - nelecb
    else:
        neleca, nelecb = nelec

    occslista = cistring._gen_occslst(range(norb), neleca)
    eri = ao2mo.restore(1, eri, norb)
    diagj = numpy.einsum('iijj->ij',eri)
    diagk = numpy.einsum('ijji->ij',eri)
    hdiag = []
    for aocc in occslista:
        bocc = aocc
        e1 = h1e[aocc,aocc].sum() + h1e[bocc,bocc].sum()
        e2 = diagj[aocc][:,aocc].sum() + diagj[aocc][:,bocc].sum() \
           + diagj[bocc][:,aocc].sum() + diagj[bocc][:,bocc].sum() \
           - diagk[aocc][:,aocc].sum() - diagk[bocc][:,bocc].sum()
        hdiag.append(e1 + e2*.5)
    return numpy.array(hdiag)

def kernel(h1e, eri, norb, nelec, ecore=0):
    return DOCI().kernel(h1e, eri, norb, nelec, ecore=ecore)

def make_rdm1(civec, norb, nelec, link_index=None):
    if isinstance(nelec, (int, numpy.integer)):
        nelecb = nelec//2
        neleca = nelec - nelecb
    else:
        neleca, nelecb = nelec
    assert(neleca == nelecb)

    if link_index is None:
        link_index = cistring.gen_linkstr_index(range(norb), neleca)
    na = cistring.num_strings(norb, neleca)
    t1 = numpy.zeros((norb,na))
    #:for str0, tab in enumerate(link_index):
    #:    for a, i, str1, sign in tab:
    #:        if a == i:
    #:            t1[i,str1] += civec[str0]
    link1 = link_index[link_index[:,:,0] == link_index[:,:,1]].reshape(na,-1,4)
    t1[link1[:,:,1],link1[:,:,2]] = civec[:,None]

    dm1 = numpy.diag(numpy.einsum('ip,p->i', t1, civec)) * 2
    return dm1

def make_rdm12(civec, norb, nelec, link_index=None, reorder=True):
    if isinstance(nelec, (int, numpy.integer)):
        nelecb = nelec//2
        neleca = nelec - nelecb
    else:
        neleca, nelecb = nelec
    assert(neleca == nelecb)

    if link_index is None:
        link_index = cistring.gen_linkstr_index(range(norb), neleca)
    na = cistring.num_strings(norb, neleca)
    t1 = numpy.zeros((norb,na))
    t2 = numpy.zeros((norb,norb,na))
    #:for str0, tab in enumerate(link_index):
    #:    for a, i, str1, sign in tab:
    #:        if a == i:
    #:            t1[i,str1] += civec[str0]
    #:        else:
    #:            t2[a,i,str1] += civec[str0]
    link1 = link_index[link_index[:,:,0] == link_index[:,:,1]].reshape(na,-1,4)
    link2 = link_index[link_index[:,:,0] != link_index[:,:,1]].reshape(na,-1,4)
    t1[link1[:,:,1],link1[:,:,2]] = civec[:,None]
    t2[link2[:,:,0],link2[:,:,1],link2[:,:,2]] = civec[:,None]

    idx = numpy.arange(norb)
    dm2 = numpy.zeros([norb]*4)
    # Assign to dm2[i,j,i,j]
    dm2[idx[:,None],idx,idx[:,None],idx] += 2 * numpy.einsum('ijp,p->ij', t2, civec)
    # Assign to dm2[i,j,j,i]
    dm2[idx[:,None],idx,idx,idx[:,None]] += 2 * numpy.einsum('ijp,ijp->ij', t2, t2)
    # Assign to dm2[i,i,j,j]
    dm2[idx[:,None],idx[:,None],idx,idx] += 4 * numpy.einsum('ip,jp->ij', t1, t1)

    dm1 = numpy.einsum('ijkk->ij', dm2) / (neleca+nelecb)

    if reorder:
        dm1, dm2 = rdm.reorder_rdm(dm1, dm2, inplace=True)
    return dm1, dm2


class DOCI(direct_spin1.FCI):
    def __init__(self, *args, **kwargs):
        direct_spin1.FCI.__init__(self, *args, **kwargs)
        self.davidson_only = True
        self.link_index = {}

    def contract_2e(self, eri, fcivec, norb, nelec, link_index=None, **kwargs):
        if link_index is None:
            link_index = self.gen_linkstr(norb, nelec)
        return contract_2e(eri, fcivec, norb, nelec, link_index)

    def make_hdiag(self, h1e, eri, norb, nelec, *args, **kwargs):
        return make_hdiag(h1e, eri, norb, nelec)

    def get_init_guess(self, norb, nelec, nroots, hdiag):
        if isinstance(nelec, (int, numpy.integer)):
            nelecb = nelec//2
            neleca = nelec - nelecb
        else:
            neleca, nelecb = nelec
        na = cistring.num_strings(norb, neleca)
        ci0 = numpy.zeros(na)
        ci0[0] = 1
        return ci0

    def kernel(self, h1e, eri, norb, nelec, ci0=None, ecore=0, **kwargs):
        if isinstance(nelec, (int, numpy.integer)):
            nelecb = nelec//2
            neleca = nelec - nelecb
        else:
            neleca, nelecb = nelec
        h2e = self.absorb_h1e(h1e, eri, norb, nelec, .5)
        h2e = ao2mo.restore(1, h2e, norb)

        hdiag = self.make_hdiag(h1e, eri, norb, nelec)
        nroots = 1
        if ci0 is None:
            ci0 = self.get_init_guess(norb, nelec, nroots, hdiag)

        def hop(c):
            return self.contract_2e(h2e, c, norb, nelec)
        precond = lambda x, e, *args: x/(hdiag-e+1e-4)
        e, c = lib.davidson(hop, ci0, precond, **kwargs)
        return e+ecore, c

    def make_rdm1(self, civec, norb, nelec, link_index=None):
        if link_index is None:
            link_index = self.gen_linkstr(norb, nelec)
        return make_rdm1(civec, norb, nelec, link_index)

    def make_rdm12(self, civec, norb, nelec, link_index=None, reorder=True):
        if link_index is None:
            link_index = self.gen_linkstr(norb, nelec)
        return make_rdm12(civec, norb, nelec, link_index, reorder)

    spin_square = None

    def gen_linkstr(self, norb, nelec):
        if (norb, nelec) not in self.link_index:
            if isinstance(nelec, (int, numpy.integer)):
                nelecb = nelec//2
                neleca = nelec - nelecb
            else:
                neleca, nelecb = nelec
            assert(neleca == nelecb)
            link_index = cistring.gen_linkstr_index(range(norb), neleca)
            self.link_index[(norb,nelec)] = link_index

        return self.link_index[(norb,nelec)]



if __name__ == '__main__':
    from functools import reduce
    from pyscf import gto
    from pyscf import scf
    from pyscf.fci import direct_spin1

    def contract_2e_ref(eri, fcivec, norb, nelec, *args, **kwargs):
        hc = direct_spin1.contract_2e(eri, numpy.diag(fcivec), norb, nelec)
        return hc.diagonal()

    norb = 6
    nelec = 3
    na = cistring.num_strings(norb, nelec)
    numpy.random.seed(2)
    eri = numpy.random.random((norb,)*4)
    eri = ao2mo.restore(1, ao2mo.restore(8, eri, norb), norb)
    ci0 = numpy.random.random(na)
    ci0 *= 1./numpy.linalg.norm(ci0)
    ci1 = contract_2e(eri, ci0, norb, (nelec,nelec))
    ci1ref = contract_2e_ref(eri, ci0, norb, (nelec,nelec))
    print(abs(ci1-ci1ref).max())

    dm1, dm2 = make_rdm12(ci0, norb, (nelec, nelec), reorder=False)
    print(numpy.einsum('ijkl,ijkl', dm2, eri) - ci1.dot(ci0))
    print(abs(dm1 - make_rdm1(ci0, norb, (nelec, nelec))).max())

    mol = gto.Mole()
    mol.verbose = 0
    mol.output = None
    mol.atom = [
        ['H', ( 1.,-1.    , 0.   )],
        ['H', ( 0.,-1.    ,-1.   )],
        ['H', ( 1.,-0.5   ,-1.   )],
        ['H', ( 0.,-0.    ,-1.   )],
        ['H', ( 1.,-0.5   , 0.   )],
        ['H', ( 0., 1.    , 1.   )],
    ]
    mol.basis = 'sto-3g'
    mol.build()

    mf = scf.RHF(mol).run()
    norb = mf.mo_coeff.shape[1]
    nelec = mol.nelectron - 2
    h1e = reduce(numpy.dot, (mf.mo_coeff.T, mf.get_hcore(), mf.mo_coeff))
    eri = ao2mo.kernel(mf._eri, mf.mo_coeff, compact=False)
    eri = eri.reshape(norb,norb,norb,norb)

    e1 = kernel(h1e, eri, norb, nelec)[0]
    print(e1, e1 - -7.9418573113478566)

    def kernel_ref(h1e, eri, norb, nelec, ecore=0, **kwargs):
        if isinstance(nelec, (int, numpy.integer)):
            nelecb = nelec//2
            neleca = nelec - nelecb
        else:
            neleca, nelecb = nelec
        h2e = direct_spin1.absorb_h1e(h1e, eri, norb, nelec, .5)
        h2e = ao2mo.restore(1, h2e, norb)
        na = cistring.num_strings(norb, neleca)
        ci0 = numpy.zeros(na)
        ci0[0] = 1

        link_index = cistring.gen_linkstr_index(range(norb), neleca)
        def hop(c):
            return contract_2e_ref(h2e, c, norb, nelec, link_index)
        hdiag = make_hdiag(h1e, eri, norb, nelec)
        precond = lambda x, e, *args: x/(hdiag-e+1e-4)
        e, c = lib.davidson(hop, ci0.reshape(-1), precond, **kwargs)
        return e+ecore
    e1ref = kernel_ref(h1e, eri, norb, nelec)
    print(e1 - e1ref)
