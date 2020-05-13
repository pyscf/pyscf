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
#

'''
Electron phonon coupling

Ref:
    arXiv:1602.04195 
'''

import numpy
from pyscf import lib
from pyscf import ao2mo
from pyscf.fci import cistring
from pyscf.fci import rdm
from pyscf.fci.direct_spin1 import _unpack_nelec

#                              site-1   ,...,site-N
#                              v             v
# ep_wfn, shape = (nstra,nstrb,nphonon+1,...,nphonon+1)
#               = (nstra,nstrb) + ([nphonon+1]*nsite)
#       For each site, {0,1,...,nphonon} gives nphonon+1 possible confs
# t for hopping, shape = (nsite,nsite)
# u
# g for the electorn-phonon coupling
# hpp for phonon-phonon interaction: (nsite,nsite)
def contract_all(t, u, g, hpp, ci0, nsite, nelec, nphonon):
    ci1  = contract_1e        (t  , ci0, nsite, nelec, nphonon)
    ci1 += contract_2e_hubbard(u  , ci0, nsite, nelec, nphonon)
    ci1 += contract_ep        (g  , ci0, nsite, nelec, nphonon)
    ci1 += contract_pp        (hpp, ci0, nsite, nelec, nphonon)
    return ci1

def make_shape(nsite, nelec, nphonon):
    neleca, nelecb = _unpack_nelec(nelec)
    na = cistring.num_strings(nsite, neleca)
    nb = cistring.num_strings(nsite, nelecb)
    return (na,nb)+(nphonon+1,)*nsite

def contract_1e(h1e, fcivec, nsite, nelec, nphonon):
    neleca, nelecb = _unpack_nelec(nelec)
    link_indexa = cistring.gen_linkstr_index(range(nsite), neleca)
    link_indexb = cistring.gen_linkstr_index(range(nsite), nelecb)
    cishape = make_shape(nsite, nelec, nphonon)

    ci0 = fcivec.reshape(cishape)
    fcinew = numpy.zeros(cishape)
    for str0, tab in enumerate(link_indexa):
        for a, i, str1, sign in tab:
            fcinew[str1] += sign * ci0[str0] * h1e[a,i]
    for str0, tab in enumerate(link_indexb):
        for a, i, str1, sign in tab:
            fcinew[:,str1] += sign * ci0[:,str0] * h1e[a,i]
    return fcinew.reshape(fcivec.shape)

# eri is a list of 2e hamiltonian (a for alpha, b for beta)
# [(aa|aa), (aa|bb), (bb|bb)]
def contract_2e(eri, fcivec, nsite, nelec, nphonon):
    neleca, nelecb = _unpack_nelec(nelec)
    link_indexa = cistring.gen_linkstr_index(range(nsite), neleca)
    link_indexb = cistring.gen_linkstr_index(range(nsite), nelecb)
    cishape = make_shape(nsite, nelec, nphonon)

    ci0 = fcivec.reshape(cishape)
    t1a = numpy.zeros((nsite,nsite)+cishape)
    t1b = numpy.zeros((nsite,nsite)+cishape)
    for str0, tab in enumerate(link_indexa):
        for a, i, str1, sign in tab:
            t1a[a,i,str1] += sign * ci0[str0]
    for str0, tab in enumerate(link_indexb):
        for a, i, str1, sign in tab:
            t1b[a,i,:,str1] += sign * ci0[:,str0]

    g2e_aa = ao2mo.restore(1, eri[0], nsite)
    g2e_ab = ao2mo.restore(1, eri[1], nsite)
    g2e_bb = ao2mo.restore(1, eri[2], nsite)
    t2a = numpy.dot(g2e_aa.reshape(nsite**2,-1), t1a.reshape(nsite**2,-1))
    t2a+= numpy.dot(g2e_ab.reshape(nsite**2,-1), t1b.reshape(nsite**2,-1))
    t2b = numpy.dot(g2e_ab.reshape(nsite**2,-1).T, t1a.reshape(nsite**2,-1))
    t2b+= numpy.dot(g2e_bb.reshape(nsite**2,-1), t1b.reshape(nsite**2,-1))

    t2a = t2a.reshape((nsite,nsite)+cishape)
    t2b = t2b.reshape((nsite,nsite)+cishape)
    fcinew = numpy.zeros(cishape)
    for str0, tab in enumerate(link_indexa):
        for a, i, str1, sign in tab:
            fcinew[str1] += sign * t2a[a,i,str0]
    for str0, tab in enumerate(link_indexb):
        for a, i, str1, sign in tab:
            fcinew[:,str1] += sign * t2b[a,i,:,str0]
    return fcinew.reshape(fcivec.shape)

def contract_2e_hubbard(u, fcivec, nsite, nelec, nphonon):
    neleca, nelecb = _unpack_nelec(nelec)
    strsa = numpy.asarray(cistring.gen_strings4orblist(range(nsite), neleca))
    strsb = numpy.asarray(cistring.gen_strings4orblist(range(nsite), nelecb))
    cishape = make_shape(nsite, nelec, nphonon)
    ci0 = fcivec.reshape(cishape)
    fcinew = numpy.zeros(cishape)

    for i in range(nsite):
        maska = (strsa & (1<<i)) > 0
        maskb = (strsb & (1<<i)) > 0
        fcinew[maska[:,None]&maskb] += u * ci0[maska[:,None]&maskb]
    return fcinew.reshape(fcivec.shape)

def slices_for(psite_id, nsite, nphonon):
    slices = [slice(None,None,None)] * (2+nsite)  # +2 for electron indices
    slices[2+psite_id] = nphonon
    return tuple(slices)
def slices_for_cre(psite_id, nsite, nphonon):
    return slices_for(psite_id, nsite, nphonon+1)
def slices_for_des(psite_id, nsite, nphonon):
    return slices_for(psite_id, nsite, nphonon-1)

# N_alpha N_beta * \sum_{p} (p^+ + p)
# N_alpha, N_beta are particle number operator, p^+ and p are phonon creation annihilation operator
def contract_ep(g, fcivec, nsite, nelec, nphonon):
    neleca, nelecb = _unpack_nelec(nelec)
    strsa = numpy.asarray(cistring.gen_strings4orblist(range(nsite), neleca))
    strsb = numpy.asarray(cistring.gen_strings4orblist(range(nsite), nelecb))
    cishape = make_shape(nsite, nelec, nphonon)
    na, nb = cishape[:2]
    ci0 = fcivec.reshape(cishape)
    fcinew = numpy.zeros(cishape)

    phonon_cre = numpy.sqrt(numpy.arange(1,nphonon+1))
    for i in range(nsite):
        maska = (strsa & (1<<i)) > 0
        maskb = (strsb & (1<<i)) > 0
        e_part = numpy.zeros((na,nb))
        e_part[maska,:] += 1
        e_part[:,maskb] += 1
        e_part[:] -= float(neleca+nelecb) / nsite
        for ip in range(nphonon):
            slices1 = slices_for_cre(i, nsite, ip)
            slices0 = slices_for    (i, nsite, ip)
            fcinew[slices1] += numpy.einsum('ij...,ij...->ij...', g*phonon_cre[ip]*e_part, ci0[slices0])
            fcinew[slices0] += numpy.einsum('ij...,ij...->ij...', g*phonon_cre[ip]*e_part, ci0[slices1])
    return fcinew.reshape(fcivec.shape)

# Contract to one phonon creation operator
def cre_phonon(fcivec, nsite, nelec, nphonon, site_id):
    cishape = make_shape(nsite, nelec, nphonon)
    ci0 = fcivec.reshape(cishape)
    fcinew = numpy.zeros(cishape)

    phonon_cre = numpy.sqrt(numpy.arange(1,nphonon+1))
    for ip in range(nphonon):
        slices1 = slices_for_cre(site_id, nsite, ip)
        slices0 = slices_for    (site_id, nsite, ip)
        fcinew[slices1] += phonon_cre[ip] * ci0[slices0]
    return fcinew.reshape(fcivec.shape)

# Contract to one phonon annihilation operator
def des_phonon(fcivec, nsite, nelec, nphonon, site_id):
    cishape = make_shape(nsite, nelec, nphonon)
    ci0 = fcivec.reshape(cishape)
    fcinew = numpy.zeros(cishape)

    phonon_cre = numpy.sqrt(numpy.arange(1,nphonon+1))
    for ip in range(nphonon):
        slices1 = slices_for_cre(site_id, nsite, ip)
        slices0 = slices_for    (site_id, nsite, ip)
        fcinew[slices0] += phonon_cre[ip] * ci0[slices1]
    return fcinew.reshape(fcivec.shape)

# Phonon-phonon coupling
def contract_pp(hpp, fcivec, nsite, nelec, nphonon):
    cishape = make_shape(nsite, nelec, nphonon)
    ci0 = fcivec.reshape(cishape)
    fcinew = numpy.zeros(cishape)

    phonon_cre = numpy.sqrt(numpy.arange(1,nphonon+1))
    t1 = numpy.zeros((nsite,)+cishape)
    for psite_id in range(nsite):
        for i in range(nphonon):
            slices1 = slices_for_cre(psite_id, nsite, i)
            slices0 = slices_for    (psite_id, nsite, i)
            t1[(psite_id,)+slices0] += ci0[slices1] * phonon_cre[i]     # annihilation

    t1 = lib.dot(hpp, t1.reshape(nsite,-1)).reshape(t1.shape)

    for psite_id in range(nsite):
        for i in range(nphonon):
            slices1 = slices_for_cre(psite_id, nsite, i)
            slices0 = slices_for    (psite_id, nsite, i)
            fcinew[slices1] += t1[(psite_id,)+slices0] * phonon_cre[i]  # creation
    return fcinew.reshape(fcivec.shape)

def make_hdiag(t, u, g, hpp, nsite, nelec, nphonon):
    neleca, nelecb = _unpack_nelec(nelec)
    link_indexa = cistring.gen_linkstr_index(range(nsite), neleca)
    link_indexb = cistring.gen_linkstr_index(range(nsite), nelecb)
    occslista = [tab[:neleca,0] for tab in link_indexa]
    occslistb = [tab[:nelecb,0] for tab in link_indexb]

    nelec_tot = neleca + nelecb

    # electron part
    cishape = make_shape(nsite, nelec, nphonon)
    hdiag = numpy.zeros(cishape)
    for ia, aocc in enumerate(occslista):
        for ib, bocc in enumerate(occslistb):
            e1 = t[aocc,aocc].sum() + t[bocc,bocc].sum()
            e2 = u * nelec_tot
            hdiag[ia,ib] = e1 + e2

    #TODO: electron-phonon part

    # phonon part
    for psite_id in range(nsite):
        for i in range(nphonon+1):
            slices0 = slices_for(psite_id, nsite, i)
            hdiag[slices0] += i+1

    return hdiag.ravel()

def kernel(t, u, g, hpp, nsite, nelec, nphonon,
           tol=1e-9, max_cycle=100, verbose=0, ecore=0, **kwargs):
    cishape = make_shape(nsite, nelec, nphonon)
    ci0 = numpy.zeros(cishape)
    ci0.__setitem__((0,0) + (0,)*nsite, 1)
    # Add noise for initial guess, remove it if problematic
    ci0[0,:] += numpy.random.random(ci0[0,:].shape) * 1e-6
    ci0[:,0] += numpy.random.random(ci0[:,0].shape) * 1e-6

    def hop(c):
        hc = contract_all(t, u, g, hpp, c, nsite, nelec, nphonon)
        return hc.reshape(-1)
    hdiag = make_hdiag(t, u, g, hpp, nsite, nelec, nphonon)
    precond = lambda x, e, *args: x/(hdiag-e+1e-4)
    e, c = lib.davidson(hop, ci0.reshape(-1), precond,
                        tol=tol, max_cycle=max_cycle, verbose=verbose,
                        **kwargs)
    return e+ecore, c


def make_rdm1e(fcivec, nsite, nelec):
    '''1-electron density matrix dm_pq = <|p^+ q|>'''
    neleca, nelecb = _unpack_nelec(nelec)
    link_indexa = cistring.gen_linkstr_index(range(nsite), neleca)
    link_indexb = cistring.gen_linkstr_index(range(nsite), nelecb)
    na = cistring.num_strings(nsite, neleca)
    nb = cistring.num_strings(nsite, nelecb)

    rdm1 = numpy.zeros((nsite,nsite))
    ci0 = fcivec.reshape(na,-1)
    for str0, tab in enumerate(link_indexa):
        for a, i, str1, sign in tab:
            rdm1[a,i] += sign * numpy.dot(ci0[str1],ci0[str0])

    ci0 = fcivec.reshape(na,nb,-1)
    for str0, tab in enumerate(link_indexb):
        for a, i, str1, sign in tab:
            rdm1[a,i] += sign * numpy.einsum('ax,ax->', ci0[:,str1],ci0[:,str0])
    return rdm1

def make_rdm12e(fcivec, nsite, nelec):
    '''1-electron and 2-electron density matrices
    dm_pq = <|p^+ q|>
    dm_{pqrs} = <|p^+ r^+ q s|>  (note 2pdm is ordered in chemist notation)
    '''
    neleca, nelecb = _unpack_nelec(nelec)
    link_indexa = cistring.gen_linkstr_index(range(nsite), neleca)
    link_indexb = cistring.gen_linkstr_index(range(nsite), nelecb)
    na = cistring.num_strings(nsite, neleca)
    nb = cistring.num_strings(nsite, nelecb)

    ci0 = fcivec.reshape(na,nb,-1)
    rdm1 = numpy.zeros((nsite,nsite))
    rdm2 = numpy.zeros((nsite,nsite,nsite,nsite))
    for str0 in range(na):
        t1 = numpy.zeros((nsite,nsite,nb)+ci0.shape[2:])
        for a, i, str1, sign in link_indexa[str0]:
            t1[i,a,:] += sign * ci0[str1,:]

        for k, tab in enumerate(link_indexb):
            for a, i, str1, sign in tab:
                t1[i,a,k] += sign * ci0[str0,str1]

        rdm1 += numpy.einsum('mp,ijmp->ij', ci0[str0], t1)
        # i^+ j|0> => <0|j^+ i, so swap i and j
        #:rdm2 += numpy.einsum('ijmp,klmp->jikl', t1, t1)
        tmp = lib.dot(t1.reshape(nsite**2,-1), t1.reshape(nsite**2,-1).T)
        rdm2 += tmp.reshape((nsite,)*4).transpose(1,0,2,3)
    rdm1, rdm2 = rdm.reorder_rdm(rdm1, rdm2, True)
    return rdm1, rdm2

def make_rdm1p(fcivec, nsite, nelec, nphonon):
    '''1-phonon density matrix dm_pq = <|p^+ q|>'''
    cishape = make_shape(nsite, nelec, nphonon)
    ci0 = fcivec.reshape(cishape)

    t1 = numpy.zeros((nsite,)+cishape)
    phonon_cre = numpy.sqrt(numpy.arange(1,nphonon+1))
    for psite_id in range(nsite):
        for i in range(nphonon):
            slices1 = slices_for_cre(psite_id, nsite, i)
            slices0 = slices_for    (psite_id, nsite, i)
            t1[(psite_id,)+slices0] += ci0[slices1] * phonon_cre[i]

    rdm1 = lib.dot(t1.reshape(nsite,-1), t1.reshape(nsite,-1).T)
    return rdm1


if __name__ == '__main__':
    nsite = 2
    nelec = 2
    nphonon = 3

    t = numpy.zeros((nsite,nsite))
    idx = numpy.arange(nsite-1)
    t[idx+1,idx] = t[idx,idx+1] = -1
    #t[:] = 0
    u = 1.5
    g = 0.5
    hpp = numpy.eye(nsite) * 1.1
    hpp[idx+1,idx] = hpp[idx,idx+1] = .1
    #hpp[:] = 0
    print('nelec = ', nelec)
    print('nphonon = ', nphonon)
    print('t =\n', t)
    print('u =', u)
    print('g =', g)
    print('hpp =\n', hpp)

#    def hop(c):
#        hc = contract_all(t, u, g, hpp, c, nsite, nelec, nphonon)
#        return hc.reshape(-1)
#    es = []
#    for nelec in ((0,0), (0,1), (1,1), (2,0), (2,1), (2,2)):
#        cishape = make_shape(nsite, nelec, nphonon)
#        n = numpy.prod(cishape)
#        hh = numpy.zeros((n,n))
#        for i in range(n):
#            z0 = numpy.zeros(n)
#            z0[i] = 1
#            hh[:,i] = hop(z0)
#        e, c = numpy.linalg.eigh(hh)
#        es.append(e)
#    print(numpy.sort(numpy.hstack(es)))
#    exit()

    es = []
    nelecs = [(ia,ib) for ia in range(nsite+1) for ib in range(ia+1)]
    for nelec in nelecs:
        e,c = kernel(t, u, g, hpp, nsite, nelec, nphonon,
                     tol=1e-10, verbose=0, nroots=1)
        print('nelec =', nelec, 'E =', e)
        es.append(e)
    es = numpy.hstack(es)
    idx = numpy.argsort(es)
    print(es[idx])

    print('\nGround state is')
    nelec = nelecs[idx[0]]
    e,c = kernel(t, u, g, hpp, nsite, nelec, nphonon,
                 tol=1e-10, verbose=0, nroots=1)
    print('nelec =', nelec, 'E =', e)
    dm1 = make_rdm1e(c, nsite, nelec)
    print('electron DM')
    print(dm1)

    dm1a, dm2 = make_rdm12e(c, nsite, nelec)
    print('check 1e DM', numpy.allclose(dm1, dm1a))
    print('check 2e DM', numpy.allclose(dm1, numpy.einsum('ijkk->ij', dm2)/(sum(nelec)-1.)))
    print('check 2e DM', numpy.allclose(dm1, numpy.einsum('kkij->ij', dm2)/(sum(nelec)-1.)))

    print('phonon DM')
    dm1 = make_rdm1p(c, nsite, nelec, nphonon)
    print(dm1)

    dm1a = numpy.empty_like(dm1)
    for i in range(nsite):
        for j in range(nsite):
            c1 = des_phonon(c, nsite, nelec, nphonon, j)
            c1 = cre_phonon(c1, nsite, nelec, nphonon, i)
            dm1a[i,j] = numpy.dot(c.ravel(), c1.ravel())
    print('check phonon DM', numpy.allclose(dm1, dm1a))

    cishape = make_shape(nsite, nelec, nphonon)
    eri = numpy.zeros((nsite,nsite,nsite,nsite))
    for i in range(nsite):
        eri[i,i,i,i] = u
    numpy.random.seed(3)
    ci0 = numpy.random.random(cishape)
    ci1 = contract_2e([eri*0,eri*.5,eri*0], ci0, nsite, nelec, nphonon)
    ci2 = contract_2e_hubbard(u, ci0, nsite, nelec, nphonon)
    print('Check contract_2e', abs(ci1-ci2).sum())
