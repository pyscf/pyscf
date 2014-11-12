#!/usr/bin/env python
# $Id$
# -*- coding: utf-8

import os
import ctypes
import numpy
import pyscf.lib
import pyscf.ao2mo
import davidson
import cistring
import rdm

_alib = os.path.join(os.path.dirname(pyscf.lib.__file__), 'libmcscf.so')
libfci = ctypes.CDLL(_alib)

def contract_1e(f1e, fcivec, norb, nelec, link_index=None):
    if link_index is None:
        link_index = cistring.gen_linkstr_index_trilidx(range(norb), nelec/2)
    na,nlink,_ = link_index.shape
    ci1 = numpy.empty((na,na))
    f1e_tril = pyscf.lib.pack_tril(f1e)
    libfci.FCIcontract_1e_ms0(f1e_tril.ctypes.data_as(ctypes.c_void_p),
                              fcivec.ctypes.data_as(ctypes.c_void_p),
                              ci1.ctypes.data_as(ctypes.c_void_p),
                              ctypes.c_int(norb), ctypes.c_int(na),
                              ctypes.c_int(nlink),
                              link_index.ctypes.data_as(ctypes.c_void_p))
    return ci1

def contract_2e(g2e, fcivec, norb, nelec, link_index=None, bufsize=1024):
    g2e = pyscf.ao2mo.restore(4, g2e, norb)
    if not g2e.flags.c_contiguous:
        g2e = g2e.copy()
    if link_index is None:
        link_index = cistring.gen_linkstr_index_trilidx(range(norb), nelec/2)
    na,nlink,_ = link_index.shape
    ci1 = numpy.empty((na,na))

    libfci.FCIcontract_2e_ms0(g2e.ctypes.data_as(ctypes.c_void_p),
                              fcivec.ctypes.data_as(ctypes.c_void_p),
                              ci1.ctypes.data_as(ctypes.c_void_p),
                              ctypes.c_int(norb), ctypes.c_int(na),
                              ctypes.c_int(nlink),
                              link_index.ctypes.data_as(ctypes.c_void_p),
                              ctypes.c_int(bufsize))
    return ci1

def make_hdiag(h1e, g2e, norb, nelec):
    g2e = pyscf.ao2mo.restore(1, g2e, norb)
    link_index = cistring.gen_linkstr_index(range(norb), nelec/2)
    na = link_index.shape[0]
    nocc = nelec / 2
    occslist = link_index[:,:nocc,0].copy('C')
    hdiag = numpy.empty(na*na)
    jdiag = numpy.einsum('iijj->ij',g2e).copy('C')
    kdiag = numpy.einsum('ijji->ij',g2e).copy('C')
    libfci.FCImake_hdiag(hdiag.ctypes.data_as(ctypes.c_void_p),
                         h1e.ctypes.data_as(ctypes.c_void_p),
                         jdiag.ctypes.data_as(ctypes.c_void_p),
                         kdiag.ctypes.data_as(ctypes.c_void_p),
                         ctypes.c_int(norb), ctypes.c_int(na),
                         ctypes.c_int(nocc),
                         occslist.ctypes.data_as(ctypes.c_void_p))
    return hdiag.ravel()

def absorb_h1e(h1e, g2e, norb, nelec):
    h2e = pyscf.ao2mo.restore(1, g2e, norb).copy()
    f1e = h1e - numpy.einsum('...,jiik->jk', .5, h2e)
    f1e = f1e * (1./nelec)
    for k in range(norb):
        h2e[k,k,:,:] += f1e
        h2e[:,:,k,k] += f1e
    return h2e

def pspace(h1e, g2e, norb, nelec, hdiag, np=400):
    g2e = pyscf.ao2mo.restore(1, g2e, norb)
    na = cistring.num_strings(norb, nelec/2)
    addr = numpy.argsort(hdiag)[:np]
# symmetrize addra/addrb
    addra = addr / na
    addrb = addr % na
    stra = numpy.array([cistring.addr2str(norb,nelec/2,ia) for ia in addra],
                       dtype=numpy.long)
    strb = numpy.array([cistring.addr2str(norb,nelec/2,ib) for ib in addrb],
                       dtype=numpy.long)
    np = len(addr)
    h0 = numpy.zeros((np,np))
    libfci.FCIpspace_h0tril(h0.ctypes.data_as(ctypes.c_void_p),
                            h1e.ctypes.data_as(ctypes.c_void_p),
                            g2e.ctypes.data_as(ctypes.c_void_p),
                            stra.ctypes.data_as(ctypes.c_void_p),
                            strb.ctypes.data_as(ctypes.c_void_p),
                            ctypes.c_int(norb), ctypes.c_int(np))

    for i in range(np):
        h0[i,i] = hdiag[addr[i]]
    h0 = pyscf.lib.hermi_triu(h0)
    return addr, h0

# be careful with single determinant initial guess. It may lead to the
# eigvalue of first davidson iter being equal to hdiag
def kernel(h1e, g2e, norb, nelec, ci0=None, eshift=.1):
    link_index = cistring.gen_linkstr_index_trilidx(range(norb), nelec/2)
    na = link_index.shape[0]
    h2e = absorb_h1e(h1e, g2e, norb, nelec) * .5
    hdiag = make_hdiag(h1e, g2e, norb, nelec)

    addr, h0 = pspace(h1e, g2e, norb, nelec, hdiag)
    pw, pv = numpy.linalg.eigh(h0)
    if len(addr) == na*na:
        ci0 = numpy.empty((na*na))
        ci0[addr] = pv[:,0]
        return pw[0], ci0.reshape(na,na)

    def precond(r, e0, x0, *args):
        #h0e0 = h0 - numpy.eye(len(addr))*(e0-eshift)
        h0e0inv = numpy.dot(pv/(pw-(e0-eshift)), pv.T)
        hdiaginv = 1/(hdiag - (e0-eshift))
        h0x0 = x0 * hdiaginv
        #h0x0[addr] = numpy.linalg.solve(h0e0, x0[addr])
        h0x0[addr] = numpy.dot(h0e0inv, x0[addr])
        h0r = r * hdiaginv
        #h0r[addr] = numpy.linalg.solve(h0e0, r[addr])
        h0r[addr] = numpy.dot(h0e0inv, r[addr])
        e1 = numpy.dot(x0, h0r) / numpy.dot(x0, h0x0)
        x1 = r - e1*x0
        #pspace_x1 = x1[addr].copy()
        x1 *= hdiaginv
# pspace (h0-e0)^{-1} cause diverging?
        #x1[addr] = numpy.linalg.solve(h0e0, pspace_x1)
        return x1
    #precond = lambda x, e, *args: x/(hdiag-(e-eshift))

    h2e = pyscf.ao2mo.restore(4, h2e, norb)
    def hop(c):
        hc = contract_2e(h2e, c, norb, nelec, link_index)
        return hc.ravel()

#TODO: check spin of initial guess
    if ci0 is None:
        # we need better initial guess
        ci0 = numpy.zeros(na*na)
        #ci0[addr] = pv[:,0]
        ci0[0] = 1
    else:
        ci0 = ci0.ravel()

    e, c = davidson.dsyev(hop, ci0, precond, tol=1e-8, lindep=1e-8)
    return e, c.reshape(na,na)

# alpha and beta 1pdm
def make_rdm1s(fcivec, norb, nelec, link_index=None):
    rdm1a = rdm.make_rdm1('FCImake_rdm1a', fcivec, fcivec,
                          norb, nelec, link_index)
    rdm1b = rdm.make_rdm1('FCImake_rdm1b', fcivec, fcivec,
                          norb, nelec, link_index)
    return rdm1a, rdm1b

# dm_pq = <|p^+ q|>
def make_rdm1(fcivec, norb, nelec, link_index=None):
    rdm1a, rdm1b = make_rdm1s(fcivec, norb, nelec, link_index)
    return rdm1a + rdm1b

# dm_pq,rs = <|p^+ q r^+ s|>
def make_rdm12(fcivec, norb, nelec, link_index=None):
    return rdm.make_rdm12('FCImake_rdm12_ms0', fcivec, fcivec,
                          norb, nelec, link_index)

# dm_pq = <I|p^+ q|J>
def trans_rdm1s(cibra, ciket, norb, nelec, link_index=None):
    if link_index is None:
        link_index = cistring.gen_linkstr_index(range(norb), nelec/2)
    rdm1a = rdm.make_rdm1('FCItrans_rdm1a', cibra, ciket,
                          norb, nelec, link_index)
    rdm1b = rdm.make_rdm1('FCItrans_rdm1b', cibra, ciket,
                          norb, nelec, link_index)
    return rdm1a, rdm1b

def trans_rdm1(cibra, ciket, norb, nelec, link_index=None):
    rdm1a, rdm1b = trans_rdm1s(cibra, ciket, norb, nelec, link_index)
    return rdm1a + rdm1b

# dm_pq,rs = <I|p^+ q r^+ s|J>
def trans_rdm12(cibra, ciket, norb, nelec, link_index=None):
    return rdm.make_rdm12('FCItrans_rdm12_ms0', cibra, ciket,
                          norb, nelec, link_index)

def energy(h1e, g2e, fcivec, norb, nelec, link_index=None):
    h2e = absorb_h1e(h1e, g2e, norb, nelec) * .5
    ci1 = contract_2e(h2e, fcivec, norb, nelec, link_index)
    return numpy.dot(fcivec.ravel(), ci1.ravel())



if __name__ == '__main__':
    import time
    from pyscf import gto
    from pyscf import scf
    from pyscf import ao2mo

    mol = gto.Mole()
    mol.verbose = 0
    mol.output = None#"out_h2o"
    mol.atom = [
        ['H', ( 1.,-1.    , 0.   )],
        ['H', ( 0.,-1.    ,-1.   )],
#        ['H', ( 1.,-0.5   ,-1.   )],
#        ['H', ( 0.,-0.5   ,-1.   )],
#        ['H', ( 0.,-0.5   ,-0.   )],
#        ['H', ( 0.,-0.    ,-1.   )],
        ['H', ( 1.,-0.5   , 0.   )],
        ['H', ( 0., 1.    , 1.   )],
    ]

    mol.basis = {'H': '6-31g'}
    mol.build()

    m = scf.RHF(mol)
    ehf = m.scf()

    norb = m.mo_coeff.shape[1]
    nelec = mol.nelectron
    h1e = reduce(numpy.dot, (m.mo_coeff.T, m.get_hcore(), m.mo_coeff))
    g2e = ao2mo.incore.general(m._eri, (m.mo_coeff,)*4, compact=False)
    g2e = ao2mo.restore(1, g2e, norb)

    print(time.clock())
    e, c = kernel(h1e, g2e, norb, nelec)
    print(e - -4.48686469648)
    print(time.clock())
