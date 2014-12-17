#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#
# FCI solver for equivalent number of alpha and beta electrons
# (requires MS=0, can be singlet, triplet, quintet, dep on init guess)
#
# Other files in the directory
# direct_ms0   MS=0, same number of alpha and beta nelectrons
# direct_spin0 singlet
# direct_spin1 arbitary number of alpha and beta electrons, based on RHF/ROHF
#              MO integrals
# direct_uhf   arbitary number of alpha and beta electrons, based on UHF
#              MO integrals
#

import os
import ctypes
import numpy
import scipy.linalg
import pyscf.lib
import pyscf.ao2mo
import cistring
import rdm
import direct_spin1

_loaderpath = os.path.dirname(pyscf.lib.__file__)
libfci = numpy.ctypeslib.load_library('libmcscf', _loaderpath)

def contract_1e(f1e, fcivec, norb, nelec, link_index=None):
    if link_index is None:
        if isinstance(nelec, int):
            neleca = nelec/2
        else:
            neleca, nelecb = nelec
            assert(neleca == nelecb)
        link_index = cistring.gen_linkstr_index_trilidx(range(norb), neleca)
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

def contract_2e(eri, fcivec, norb, nelec, link_index=None):
    eri = pyscf.ao2mo.restore(4, eri, norb)
    if not eri.flags.c_contiguous:
        eri = eri.copy()
    if link_index is None:
        if isinstance(nelec, int):
            neleca = nelec/2
        else:
            neleca, nelecb = nelec
            assert(neleca == nelecb)
        link_index = cistring.gen_linkstr_index_trilidx(range(norb), neleca)
    na,nlink,_ = link_index.shape
    ci1 = numpy.empty((na,na))

    libfci.FCIcontract_2e_ms0(eri.ctypes.data_as(ctypes.c_void_p),
                              fcivec.ctypes.data_as(ctypes.c_void_p),
                              ci1.ctypes.data_as(ctypes.c_void_p),
                              ctypes.c_int(norb), ctypes.c_int(na),
                              ctypes.c_int(nlink),
                              link_index.ctypes.data_as(ctypes.c_void_p))
    return ci1

def absorb_h1e(*args, **kwargs):
    return direct_spin1.absorb_h1e(*args, **kwargs)

def make_hdiag(h1e, eri, norb, nelec):
    if isinstance(nelec, int):
        neleca = nelec/2
    else:
        neleca, nelecb = nelec
        assert(neleca == nelecb)
    eri = pyscf.ao2mo.restore(1, eri, norb)
    link_index = cistring.gen_linkstr_index(range(norb), neleca)
    na = link_index.shape[0]
    occslist = link_index[:,:neleca,0].copy('C')
    hdiag = numpy.empty((na,na))
    jdiag = numpy.einsum('iijj->ij',eri).copy('C')
    kdiag = numpy.einsum('ijji->ij',eri).copy('C')
    libfci.FCImake_hdiag(hdiag.ctypes.data_as(ctypes.c_void_p),
                         h1e.ctypes.data_as(ctypes.c_void_p),
                         jdiag.ctypes.data_as(ctypes.c_void_p),
                         kdiag.ctypes.data_as(ctypes.c_void_p),
                         ctypes.c_int(norb), ctypes.c_int(na),
                         ctypes.c_int(neleca),
                         occslist.ctypes.data_as(ctypes.c_void_p))
# symmetrize hdiag to reduce numerical error
    hdiag = pyscf.lib.transpose_sum(hdiag, inplace=True) * .5
    return hdiag.ravel()

def pspace(h1e, eri, norb, nelec, hdiag, np=400):
    if isinstance(nelec, int):
        neleca = nelec/2
    else:
        neleca, nelecb = nelec
        assert(neleca == nelecb)
    eri = pyscf.ao2mo.restore(1, eri, norb)
    na = cistring.num_strings(norb, neleca)
    addr = numpy.argsort(hdiag)[:np]
# symmetrize addra/addrb
    addra = addr / na
    addrb = addr % na
    stra = numpy.array([cistring.addr2str(norb,neleca,ia) for ia in addra],
                       dtype=numpy.long)
    strb = numpy.array([cistring.addr2str(norb,neleca,ib) for ib in addrb],
                       dtype=numpy.long)
    np = len(addr)
    h0 = numpy.zeros((np,np))
    libfci.FCIpspace_h0tril(h0.ctypes.data_as(ctypes.c_void_p),
                            h1e.ctypes.data_as(ctypes.c_void_p),
                            eri.ctypes.data_as(ctypes.c_void_p),
                            stra.ctypes.data_as(ctypes.c_void_p),
                            strb.ctypes.data_as(ctypes.c_void_p),
                            ctypes.c_int(norb), ctypes.c_int(np))

    for i in range(np):
        h0[i,i] = hdiag[addr[i]]
    h0 = pyscf.lib.hermi_triu(h0)
    return addr, h0

# be careful with single determinant initial guess. It may lead to the
# eigvalue of first davidson iter being equal to hdiag
def kernel(h1e, eri, norb, nelec, ci0=None, eshift=.1, **kwargs):
    if isinstance(nelec, int):
        neleca = nelec/2
    else:
        neleca, nelecb = nelec
        assert(neleca == nelecb)
    link_index = cistring.gen_linkstr_index_trilidx(range(norb), neleca)
    na = link_index.shape[0]
    h2e = direct_spin1.absorb_h1e(h1e, eri, norb, nelec, .5)
    hdiag = make_hdiag(h1e, eri, norb, nelec)

    addr, h0 = pspace(h1e, eri, norb, nelec, hdiag)
    pw, pv = scipy.linalg.eigh(h0)
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

    e, c = pyscf.lib.davidson(hop, ci0, precond, tol=1e-8, lindep=1e-8)
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
# need call reorder_rdm for this rdm2
def make_rdm12(fcivec, norb, nelec, link_index=None):
    return rdm.make_rdm12('FCImake_rdm12_ms0', fcivec, fcivec,
                          norb, nelec, link_index)

# dm_pq = <I|p^+ q|J>
def trans_rdm1s(cibra, ciket, norb, nelec, link_index=None):
    if link_index is None:
        if isinstance(nelec, int):
            neleca = nelec/2
        else:
            neleca, nelecb = nelec
            assert(neleca == nelecb)
        link_index = cistring.gen_linkstr_index(range(norb), neleca)
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

def energy(h1e, eri, fcivec, norb, nelec, link_index=None):
    h2e = direct_spin1.absorb_h1e(h1e, eri, norb, nelec, .5)
    ci1 = contract_2e(h2e, fcivec, norb, nelec, link_index)
    return numpy.dot(fcivec.ravel(), ci1.ravel())


###############################################################
# direct-CI driver
###############################################################

def kernel_ms0(fci, h1e, eri, norb, nelec, ci0=None):
    if isinstance(nelec, int):
        neleca = nelec/2
    else:
        neleca, nelecb = nelec
        assert(neleca == nelecb)
    link_index = cistring.gen_linkstr_index_trilidx(range(norb), neleca)
    na = link_index.shape[0]
    h2e = fci.absorb_h1e(h1e, eri, norb, nelec, .5)
    hdiag = fci.make_hdiag(h1e, eri, norb, nelec)

    addr, h0 = fci.pspace(h1e, eri, norb, nelec, hdiag)
    pw, pv = scipy.linalg.eigh(h0)
    if len(addr) == na*na:
        ci0 = numpy.empty((na*na))
        ci0[addr] = pv[:,0]
        return pw[0], ci0.reshape(na,na)

    precond = fci.make_precond(hdiag, pw, pv, addr)

    def hop(c):
        hc = fci.contract_2e(h2e, c, norb, nelec, link_index)
        return hc.ravel()

#TODO: check spin of initial guess
    if ci0 is None:
        # we need better initial guess
        ci0 = numpy.zeros(na*na)
        #ci0[addr] = pv[:,0]
        ci0[0] = 1
    else:
        ci0 = ci0.ravel()

    #e, c = pyscf.lib.davidson(hop, ci0, precond, tol=fci.tol, lindep=fci.lindep)
    e, c = fci.eig(hop, ci0, precond)
    return e, c.reshape(na,na)


class FCISolver(direct_spin1.FCISolver):

    def make_hdiag(self, h1e, eri, norb, nelec):
        return make_hdiag(h1e, eri, norb, nelec)

    def pspace(self, h1e, eri, norb, nelec, hdiag, np=400):
        return pspace(h1e, eri, norb, nelec, hdiag, np)

    def contract_1e(self, f1e, fcivec, norb, nelec, link_index=None, **kwargs):
        return contract_1e(f1e, fcivec, norb, nelec, link_index, **kwargs)

    def contract_2e(self, eri, fcivec, norb, nelec, link_index=None, **kwargs):
        return contract_2e(eri, fcivec, norb, nelec, link_index, **kwargs)

    def kernel(self, h1e, eri, norb, nelec, ci0=None, **kwargs):
        self.mol.check_sanity(self)
        return kernel_ms0(self, h1e, eri, norb, nelec, ci0)

    def make_rdm1s(self, fcivec, norb, nelec, link_index=None, **kwargs):
        return make_rdm1s(fcivec, norb, nelec, link_index)

    def make_rdm1(self, fcivec, norb, nelec, link_index=None, **kwargs):
        return make_rdm1(fcivec, norb, nelec, link_index)

    def make_rdm12(self, fcivec, norb, nelec, link_index=None, **kwargs):
        return make_rdm12(fcivec, norb, nelec, link_index)

    def trans_rdm1s(self, cibra, ciket, norb, nelec, link_index=None, **kwargs):
        return trans_rdm1s(cibra, ciket, norb, nelec, link_index)

    def trans_rdm1(self, cibra, ciket, norb, nelec, link_index=None, **kwargs):
        return trans_rdm1(cibra, ciket, norb, nelec, link_index)

    def trans_rdm12(self, cibra, ciket, norb, nelec, link_index=None, **kwargs):
        return trans_rdm12(cibra, ciket, norb, nelec, link_index)



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

    cis = FCISolver(mol)
    norb = m.mo_coeff.shape[1]
    nelec = mol.nelectron
    h1e = reduce(numpy.dot, (m.mo_coeff.T, m.get_hcore(), m.mo_coeff))
    eri = ao2mo.incore.general(m._eri, (m.mo_coeff,)*4, compact=False)

    print(time.clock())
    e, c = cis.kernel(h1e, eri, norb, nelec)
    print(e - -4.48686469648)
    print(time.clock())
