#!/usr/bin/env python
#
# File: direct_spin0.py
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

import math
import numpy
from pyscf import lib
from pyscf import ao2mo
from pyscf.lib import _mcscf
import davidson
import cistring


def contract_1e(f1e, fcivec, norb, nelec, link_index=None):
    if link_index is None:
        link_index = cistring.gen_linkstr_index(range(norb), nelec/2)
    ci1 = _mcscf.contract_1e_spin0(f1e, fcivec, norb, link_index)
    return lib.transpose_sum(ci1, inplace=True)

def contract_2e(g2e, fcivec, norb, nelec, link_index=None):
    g2e = ao2mo.restore(4, g2e, norb)
    if not g2e.flags.c_contiguous:
        g2e = g2e.copy()
    if link_index is None:
        link_index = cistring.gen_linkstr_index(range(norb), nelec/2)
#FIXME: symmetrization is necessary, to reduce numerical error?
    na = link_index.shape[0]
    fcivec = lib.transpose_sum(fcivec.reshape(na,na)) * .5
    ci1 = _mcscf.contract_2e_spin0_omp(g2e, fcivec, norb, link_index)
    return lib.transpose_sum(ci1, inplace=True)

def make_hdiag(h1e, g2e, norb, nelec, link_index=None):
    g2e = ao2mo.restore(1, g2e, norb)
    if link_index is None:
        link_index = cistring.gen_linkstr_index(range(norb), nelec/2)
    hdiag = _mcscf.make_hdiag(h1e, g2e, norb, nelec, link_index)
    na = link_index.shape[0]
    hdiag = lib.transpose_sum(hdiag.reshape(na,na), inplace=True) * .5
    return hdiag.reshape(-1)

def absorb_h1e(h1e, g2e, norb, nelec):
    h2e = ao2mo.restore(1, g2e, norb).copy()
    f1e = h1e - numpy.einsum('...,jiik->jk', .5, h2e)
    f1e = f1e * (1./nelec)
    for k in range(norb):
        h2e[k,k,:,:] += f1e
        h2e[:,:,k,k] += f1e
    return h2e

def kernel(h1e, g2e, norb, nelec, ci0=None, eshift=1e-8):
    link_index = cistring.gen_linkstr_index(range(norb), nelec/2)
    na = link_index.shape[0]
    h2e = absorb_h1e(h1e, g2e, norb, nelec) * .5
    hdiag = make_hdiag(h1e, g2e, norb, nelec, link_index)
    precond = lambda x, e: x/(hdiag-(e-eshift))

    h2e = ao2mo.restore(4, h2e, norb)
    def hop(c):
        hc = contract_2e(h2e, c, norb, nelec, link_index)
        return hc.reshape(-1)

    if ci0 is None:
        # we need better initial guess
        # single determinant initial guess may cause precond diverge
        ci0 = numpy.zeros(na*na)
        ci0[0] = 1
        ci0[-1] = -1e-3
    else:
        ci0 = ci0.reshape(-1)

    e, c = davidson.dsyev(hop, ci0, precond, tol=1e-9)
    return e, lib.transpose_sum(c.reshape(na,na)) * .5

# dm_pq = <|p^+ q|>
def make_rdm1(fcivec, norb, nelec, link_index=None):
    if link_index is None:
        link_index = cistring.gen_linkstr_index(range(norb), nelec/2)
    return _mcscf.make_rdm1_spin0_o3(fcivec, norb, link_index)

# dm_pq,rs = <|p^+ q r^+ s|>
# dm_pq,rs = dm_sr,qp;  dm_qp,rs = dm_rs,qp
# need call reorder_rdm for this rdm2 to get standard 2pdm
def _make_rdm2(fcivec, norb, nelec, link_index=None):
    if link_index is None:
        link_index = cistring.gen_linkstr_index(range(norb), nelec/2)
    return _mcscf.make_rdm2_o2(fcivec, norb, link_index)

def make_rdm12_o1(fcivec, norb, nelec, link_index=None):
    if link_index is None:
        link_index = cistring.gen_linkstr_index(range(norb), nelec/2)
    na = link_index.shape[0]
    fcivec = lib.transpose_sum(fcivec.reshape(na,na)) * .5

    rdm1 = numpy.zeros((norb,norb))
    rdm2 = numpy.zeros((norb,norb,norb,norb))
    t1 = numpy.zeros((na,na,norb,norb))
    t1d = numpy.zeros((na,norb,norb))
    for str0, tab in enumerate(link_index):
        for a, i, str1, sign in tab:
            for k in range(str0):
                t1[str0,k,i,a] += sign * fcivec[str1,k]
            t1d[str0,i,a] += sign * fcivec[str1,str0]

        for k in range(str0):
            tab = link_index[k]
            for a, i, str1, sign in tab:
                t1[str0,k,i,a] += sign * fcivec[str0,str1]

    rdm1 += numpy.einsum('mn,mnij->ij', fcivec, t1)
    rdm1 += numpy.einsum('m,mij->ij', fcivec.diagonal(), t1d)
    # i^+ j|0> => <0|j^+ i, so swap i and j
    rdm2 += numpy.einsum('mnij,mnkl->jikl', t1, t1) * 2
    rdm2 += numpy.einsum('mij,mkl->jikl', t1d, t1d) * 4
    rdm1 = rdm1 + rdm1.T
    #print 't1',abs(t1.transpose(1,0,2,3)-t1).sum()
    #print 'rdm2',abs(rdm2-rdm2.transpose(3,2,1,0)).sum()
    #print 'rdm2',abs(rdm2.transpose(1,0,2,3)-rdm2.transpose(2,3,1,0)).sum()
    return _mcscf.reorder_rdm(rdm1, rdm2, True)

def make_rdm12_o2(fcivec, norb, nelec, link_index=None):
    if link_index is None:
        link_index = cistring.gen_linkstr_index(range(norb), nelec/2)
    na = link_index.shape[0]
    fcivec = lib.transpose_sum(fcivec.reshape(na,na)) * .5

    rdm1 = numpy.zeros((norb,norb))
    rdm2 = numpy.zeros((norb,norb,norb,norb))
    for str0, tab in enumerate(link_index):
        t1 = numpy.zeros((str0+1,norb,norb))
        for a, i, str1, sign in tab:
            for k in range(str0+1):
                t1[k,i,a] += sign * fcivec[str1,k]

        for k in range(str0):
            tab = link_index[k]
            for a, i, str1, sign in tab:
                t1[k,i,a] += sign * fcivec[str0,str1]

        rdm1 += numpy.einsum('m,mij->ij', fcivec[str0,:str0+1], t1)
        # i^+ j|0> => <0|j^+ i, so swap i and j
        rdm2 += numpy.einsum('mij,mkl->jikl', t1, t1)
        rdm2 += numpy.einsum('ij,kl->jikl', t1[str0], t1[str0])
    rdm1 = rdm1 + rdm1.T
    rdm2 = rdm2 + rdm2.transpose(3,2,1,0)
    return _mcscf.reorder_rdm(rdm1, rdm2, True)

def make_rdm12(fcivec, norb, nelec, link_index=None):
    if link_index is None:
        link_index = cistring.gen_linkstr_index(range(norb), nelec/2)
    na = link_index.shape[0]
    fcivec = lib.transpose_sum(fcivec.reshape(na,na)) * .5
    return _mcscf.make_rdm12_spin0_omp(fcivec, norb, link_index)

# dm_pq = <I|p^+ q|J>
def trans_rdm1_o1(cibra, ciket, norb, nelec, link_index=None):
    if link_index is None:
        link_index = cistring.gen_linkstr_index(range(norb), nelec/2)
    na = link_index.shape[0]
    ciket = ciket.reshape(na,na)
    cibra = cibra.reshape(na,na)

    rdm1 = numpy.zeros((norb,norb))
    for str0 in range(na):
        t1ket = numpy.zeros((str0+1,norb,norb))
        for a, i, str1, sign in link_index[str0]:
            for k in range(str0+1):
                t1ket[k,i,a] += sign * ciket[str1,k]
        for k in range(str0):
            for a, i, str1, sign in link_index[k]:
                t1ket[k,i,a] += sign * ciket[str0,str1]
        rdm1 += numpy.einsum('m,mij->ij', cibra[str0,:str0+1], t1ket)
    return rdm1*2

def trans_rdm1(cibra, ciket, norb, nelec, link_index=None):
    if link_index is None:
        link_index = cistring.gen_linkstr_index(range(norb), nelec/2)
    return _mcscf.trans_rdm1_spin0_o2(cibra, ciket, norb, link_index)

# dm_pq,rs = <I|p^+ q r^+ s|J>
def _trans_rdm2_o1(cibra, ciket, norb, nelec, link_index=None):
    if link_index is None:
        link_index = cistring.gen_linkstr_index(range(norb), nelec/2)
    na = link_index.shape[0]
    cibra = cibra.reshape(na,na)
    ciket = ciket.reshape(na,na)

    rdm2 = numpy.zeros((norb,norb,norb,norb))
    for str0 in range(na):
        t1bra = numpy.zeros((str0+1,norb,norb))
        t1ket = numpy.zeros((str0+1,norb,norb))
        for k in range(str0+1):
            for a, i, str1, sign in link_index[str0]:
                t1bra[k,i,a] += sign * cibra[str1,k]
                t1ket[k,i,a] += sign * ciket[str1,k]
        for k in range(str0):
            for a, i, str1, sign in link_index[k]:
                t1bra[k,i,a] += sign * cibra[str0,str1]
                t1ket[k,i,a] += sign * ciket[str0,str1]
        rdm2 += numpy.einsum('mij,mkl->jikl', t1bra[:str0], t1ket[:str0])
        rdm2 += numpy.einsum('ij,kl->jikl', t1bra[str0], t1ket[str0])*2
    return rdm2 * 2

def _trans_rdm2(cibra, ciket, norb, nelec, link_index=None):
    if link_index is None:
        link_index = cistring.gen_linkstr_index(range(norb), nelec/2)
    return _mcscf.trans_rdm2_o2(cibra, ciket, norb, link_index)

def trans_rdm12(cibra, ciket, norb, nelec, link_index=None):
    if link_index is None:
        link_index = cistring.gen_linkstr_index(range(norb), nelec/2)
    return _mcscf.trans_rdm12_spin0_omp(cibra, ciket, norb, link_index)

def energy(h1e, g2e, fcivec, norb, nelec, link_index=None):
    h2e = absorb_h1e(h1e, g2e, norb, nelec) * .5
    ci1 = contract_2e(h2e, fcivec, norb, nelec, link_index=None)
    return numpy.dot(fcivec.reshape(-1), ci1.reshape(-1))



if __name__ == '__main__':
    from pyscf import gto
    from pyscf import scf
    from pyscf import ao2mo

    mol = gto.Mole()
    mol.verbose = 0
    mol.output = None#"out_h2o"
    mol.atom = [
        ['H', ( 1.,-1.    , 0.   )],
        ['H', ( 0.,-1.    ,-1.   )],
        ['H', ( 1.,-0.5   ,-1.   )],
        ['H', ( 0.,-0.5   ,-1.   )],
        ['H', ( 0.,-0.5   ,-0.   )],
        ['H', ( 0.,-0.    ,-1.   )],
        ['H', ( 1.,-0.5   , 0.   )],
        ['H', ( 0., 1.    , 1.   )],
    ]

    mol.basis = {'H': 'sto-3g'}
    mol.build()

    m = scf.RHF(mol)
    ehf = m.scf()

    norb = m.mo_coeff.shape[1]
    nelec = mol.nelectron
    h1e = reduce(numpy.dot, (m.mo_coeff.T, m.get_hcore(), m.mo_coeff))
    g2e = ao2mo.incore.general(m._eri, (m.mo_coeff,)*4, compact=False)
    e, c = kernel(h1e, g2e, norb, nelec)
    rdm1, rdm2 = make_rdm12(c, norb, nelec)
    print e, e - -15.9977886547
