#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#
# FCI solver for arbitary number of alpha and beta electrons.  The hamiltonian
# requires spacial part of the integrals (RHF/ROHF MO integrals).  This solver
# can be used to compute doublet, triplet,...
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
import pyscf.lib
import pyscf.ao2mo
import davidson
import cistring
import rdm

_loaderpath = os.path.dirname(pyscf.lib.__file__)
libfci = numpy.ctypeslib.load_library('libmcscf', _loaderpath)

def contract_1e(f1e, fcivec, norb, nelec, link_index=None):
    if link_index is None:
        if isinstance(nelec, int):
            nelecb = nelec/2
            neleca = nelec - nelecb
        else:
            neleca, nelecb = nelec
        link_indexa = cistring.gen_linkstr_index_trilidx(range(norb), neleca)
        link_indexb = cistring.gen_linkstr_index_trilidx(range(norb), nelecb)
    else:
        link_indexa, link_indexb = link_index

    na, nlinka = link_indexa.shape[:2]
    nb, nlinkb = link_indexb.shape[:2]
    f1e_tril = pyscf.lib.pack_tril(f1e)
    ci1 = numpy.zeros((na,nb))
    libfci.FCIcontract_a_1e(f1e_tril.ctypes.data_as(ctypes.c_void_p),
                            fcivec.ctypes.data_as(ctypes.c_void_p),
                            ci1.ctypes.data_as(ctypes.c_void_p),
                            ctypes.c_int(norb),
                            ctypes.c_int(na), ctypes.c_int(nb),
                            ctypes.c_int(nlinka), ctypes.c_int(nlinkb),
                            link_indexa.ctypes.data_as(ctypes.c_void_p),
                            link_indexb.ctypes.data_as(ctypes.c_void_p))
    libfci.FCIcontract_b_1e(f1e_tril.ctypes.data_as(ctypes.c_void_p),
                            fcivec.ctypes.data_as(ctypes.c_void_p),
                            ci1.ctypes.data_as(ctypes.c_void_p),
                            ctypes.c_int(norb),
                            ctypes.c_int(na), ctypes.c_int(nb),
                            ctypes.c_int(nlinka), ctypes.c_int(nlinkb),
                            link_indexa.ctypes.data_as(ctypes.c_void_p),
                            link_indexb.ctypes.data_as(ctypes.c_void_p))
    return ci1

# the input fcivec should be symmetrized
def contract_2e(eri, fcivec, norb, nelec, link_index=None, bufsize=1024):
    eri = pyscf.ao2mo.restore(4, eri, norb)
    if not eri.flags.c_contiguous:
        eri = eri.copy()
    if link_index is None:
        if isinstance(nelec, int):
            nelecb = nelec/2
            neleca = nelec - nelecb
        else:
            neleca, nelecb = nelec
        link_indexa = cistring.gen_linkstr_index_trilidx(range(norb), neleca)
        link_indexb = cistring.gen_linkstr_index_trilidx(range(norb), nelecb)
    else:
        link_indexa, link_indexb = link_index

    na, nlinka = link_indexa.shape[:2]
    nb, nlinkb = link_indexb.shape[:2]
    fcivec = fcivec.reshape(na,nb)
    ci1 = numpy.empty_like(fcivec)

    libfci.FCIcontract_rhf2e_spin1(eri.ctypes.data_as(ctypes.c_void_p),
                                   fcivec.ctypes.data_as(ctypes.c_void_p),
                                   ci1.ctypes.data_as(ctypes.c_void_p),
                                   ctypes.c_int(norb),
                                   ctypes.c_int(na), ctypes.c_int(nb),
                                   ctypes.c_int(nlinka), ctypes.c_int(nlinkb),
                                   link_indexa.ctypes.data_as(ctypes.c_void_p),
                                   link_indexb.ctypes.data_as(ctypes.c_void_p),
                                   ctypes.c_int(bufsize))
    return ci1

def make_hdiag(h1e, eri, norb, nelec):
    if isinstance(nelec, int):
        nelecb = nelec/2
        neleca = nelec - nelecb
    else:
        neleca, nelecb = nelec
    eri = pyscf.ao2mo.restore(1, eri, norb)
    link_indexa = cistring.gen_linkstr_index(range(norb), neleca)
    link_indexb = cistring.gen_linkstr_index(range(norb), nelecb)
    na = link_indexa.shape[0]
    nb = link_indexb.shape[0]

    occslista = link_indexa[:,:neleca,0].copy('C')
    occslistb = link_indexb[:,:nelecb,0].copy('C')
    hdiag = numpy.empty(na*nb)
    jdiag = numpy.einsum('iijj->ij',eri).copy('C')
    kdiag = numpy.einsum('ijji->ij',eri).copy('C')
    libfci.FCImake_hdiag_uhf(hdiag.ctypes.data_as(ctypes.c_void_p),
                             h1e.ctypes.data_as(ctypes.c_void_p),
                             h1e.ctypes.data_as(ctypes.c_void_p),
                             jdiag.ctypes.data_as(ctypes.c_void_p),
                             jdiag.ctypes.data_as(ctypes.c_void_p),
                             jdiag.ctypes.data_as(ctypes.c_void_p),
                             kdiag.ctypes.data_as(ctypes.c_void_p),
                             kdiag.ctypes.data_as(ctypes.c_void_p),
                             ctypes.c_int(norb),
                             ctypes.c_int(na), ctypes.c_int(nb),
                             ctypes.c_int(neleca), ctypes.c_int(nelecb),
                             occslista.ctypes.data_as(ctypes.c_void_p),
                             occslistb.ctypes.data_as(ctypes.c_void_p))
    return numpy.array(hdiag)

def absorb_h1e(h1e, eri, norb, nelec, fac=1):
    if not isinstance(nelec, int):
        nelec = sum(nelec)
    h2e = pyscf.ao2mo.restore(1, eri, norb).copy()
    f1e = h1e - numpy.einsum('...,jiik->jk', .5, h2e)
    f1e = f1e * (1./nelec)
    for k in range(norb):
        h2e[k,k,:,:] += f1e
        h2e[:,:,k,k] += f1e
    return pyscf.ao2mo.restore(4, h2e, norb) * fac

def pspace(h1e, eri, norb, nelec, hdiag, np=400):
    if isinstance(nelec, int):
        nelecb = nelec/2
        neleca = nelec - nelecb
    else:
        neleca, nelecb = nelec
    eri = pyscf.ao2mo.restore(1, eri, norb)
    na = cistring.num_strings(norb, neleca)
    nb = cistring.num_strings(norb, nelecb)
    addr = numpy.argsort(hdiag)[:np]
    addra = addr / nb
    addrb = addr % nb
    stra = numpy.array([cistring.addr2str(norb,neleca,ia) for ia in addra],
                       dtype=numpy.long)
    strb = numpy.array([cistring.addr2str(norb,nelecb,ib) for ib in addrb],
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
def kernel(h1e, eri, norb, nelec, ci0=None, eshift=.001, **kwargs):
    if isinstance(nelec, int):
        nelecb = nelec/2
        neleca = nelec - nelecb
        nelec = (neleca, nelecb)
    else:
        neleca, nelecb = nelec
    link_indexa = cistring.gen_linkstr_index_trilidx(range(norb), neleca)
    link_indexb = cistring.gen_linkstr_index_trilidx(range(norb), nelecb)
    na = link_indexa.shape[0]
    nb = link_indexb.shape[0]
    h2e = absorb_h1e(h1e, eri, norb, nelec, .5)
    hdiag = make_hdiag(h1e, eri, norb, nelec)

    addr, h0 = pspace(h1e, eri, norb, nelec, hdiag)
    pw, pv = numpy.linalg.eigh(h0)
    if len(addr) == na*nb:
        ci0 = numpy.empty((na*nb))
        ci0[addr] = pv[:,0]
        return pw[0], ci0.reshape(na,nb)

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

    def hop(c):
        hc = contract_2e(h2e, c, norb, nelec, (link_indexa,link_indexb))
        return hc.ravel()

    if ci0 is None:
        ci0 = numpy.zeros(na*nb)
        ci0[0] = 1
    else:
        ci0 = ci0.ravel()

    e, c = davidson.dsyev(hop, ci0, precond, tol=1e-8, lindep=1e-8)
    return e, c.reshape(na,nb)

def energy(h1e, eri, fcivec, norb, nelec, link_index=None):
    h2e = absorb_h1e(h1e, eri, norb, nelec, .5)
    ci1 = contract_2e(h2e, fcivec, norb, nelec, link_index)
    return numpy.dot(fcivec.reshape(-1), ci1.reshape(-1))


# dm_pq = <|p^+ q|>
def make_rdm1s(fcivec, norb, nelec, link_index=None):
    if link_index is None:
        link_indexa = cistring.gen_linkstr_index(range(norb), nelec[0])
        link_indexb = cistring.gen_linkstr_index(range(norb), nelec[1])
        link_index = (link_indexa, link_indexb)
    rdm1a = rdm.make_rdm1_spin1('FCImake_rdm1a', fcivec, fcivec,
                                norb, nelec, link_index)
    rdm1b = rdm.make_rdm1_spin1('FCImake_rdm1b', fcivec, fcivec,
                                norb, nelec, link_index)
    return rdm1a, rdm1b

# spacial part of DM, dm_pq = <|p^+ q|>
def make_rdm1(fcivec, norb, nelec, link_index=None):
    rdm1a, rdm1b = make_rdm1s(fcivec, norb, nelec, link_index)
    return rdm1a + rdm1b

def make_rdm12s(fcivec, norb, nelec, link_index=None):
    dm1a, dm2aa = rdm.make_rdm12_spin1('FCIrdm12kern_a', fcivec, fcivec,
                                       norb, nelec, link_index, 1)
    dm1b, dm2bb = rdm.make_rdm12_spin1('FCIrdm12kern_b', fcivec, fcivec,
                                       norb, nelec, link_index, 1)
    _, dm2ab = rdm.make_rdm12_spin1('FCItdm12kern_ab', fcivec, fcivec,
                                    norb, nelec, link_index, 0)
    dm1a, dm2aa = rdm.reorder_rdm(dm1a, dm2aa, inplace=True)
    dm1b, dm2bb = rdm.reorder_rdm(dm1b, dm2bb, inplace=True)
    return (dm1a, dm1b), (dm2aa, dm2ab, dm2bb)

def make_rdm12(fcivec, norb, nelec, link_index=None):
    (dm1a, dm1b), (dm2aa, dm2ab, dm2bb) = \
            make_rdm12s(fcivec, norb, nelec, link_index)
    return dm1a+dm1b, dm2aa+dm2ab+dm2ab.transpose(2,3,0,1)+dm2bb

def trans_rdm1s(cibra, ciket, norb, nelec, link_index=None):
    rdm1a = rdm.make_rdm1_spin1('FCItrans_rdm1a', cibra, ciket,
                                norb, nelec, link_index)
    rdm1b = rdm.make_rdm1_spin1('FCItrans_rdm1b', cibra, ciket,
                                norb, nelec, link_index)
    return rdm1a, rdm1b

# spacial part of DM
def trans_rdm1(cibra, ciket, norb, nelec, link_index=None):
    rdm1a, rdm1b = trans_rdm1s(cibra, ciket, norb, nelec, link_index)
    return rdm1a + rdm1b

def trans_rdm12s(cibra, ciket, norb, nelec, link_index=None):
    dm1a, dm2aa = rdm.make_rdm12_spin1('FCItdm12kern_a', cibra, ciket,
                                       norb, nelec, link_index, 0)
    dm1b, dm2bb = rdm.make_rdm12_spin1('FCItdm12kern_b', cibra, ciket,
                                       norb, nelec, link_index, 0)
    _, dm2ab = rdm.make_rdm12_spin1('FCItdm12kern_ab', cibra, ciket,
                                    norb, nelec, link_index, 0)
    _, dm2ba = rdm.make_rdm12_spin1('FCItdm12kern_ab', ciket, cibra,
                                    norb, nelec, link_index, 0)
    dm2ba = dm2ba.transpose(3,2,1,0)
    dm1a, dm2aa = rdm.reorder_rdm(dm1a, dm2aa, inplace=True)
    dm1b, dm2bb = rdm.reorder_rdm(dm1b, dm2bb, inplace=True)
    return (dm1a, dm1b), (dm2aa, dm2ab, dm2ba, dm2bb)

def trans_rdm12(cibra, ciket, norb, nelec, link_index=None):
    (dm1a, dm1b), (dm2aa, dm2ab, dm2ba, dm2bb) = \
            trans_rdm12s(cibra, ciket, norb, nelec, link_index)
    return dm1a+dm1b, dm2aa+dm2ab+dm2ba+dm2bb



###############################################################
# direct-CI driver
###############################################################

def kernel_ms1(fci, h1e, eri, norb, nelec, ci0=None):
    if isinstance(nelec, int):
        nelecb = nelec/2
        neleca = nelec - nelecb
        nelec = (neleca, nelecb)
    else:
        neleca, nelecb = nelec
    link_indexa = cistring.gen_linkstr_index_trilidx(range(norb), neleca)
    link_indexb = cistring.gen_linkstr_index_trilidx(range(norb), nelecb)
    na = link_indexa.shape[0]
    nb = link_indexb.shape[0]
    h2e = fci.absorb_h1e(h1e, eri, norb, nelec, .5)
    hdiag = fci.make_hdiag(h1e, eri, norb, nelec)

    addr, h0 = fci.pspace(h1e, eri, norb, nelec, hdiag)
    pw, pv = numpy.linalg.eigh(h0)
    if len(addr) == na*nb:
        ci0 = numpy.empty((na*nb))
        ci0[addr] = pv[:,0]
        return pw[0], ci0.reshape(na,nb)

    precond = fci.make_precond(hdiag, pw, pv, addr)

    def hop(c):
        hc = fci.contract_2e(h2e, c, norb, nelec, (link_indexa,link_indexb))
        return hc.ravel()

    if ci0 is None:
        ci0 = numpy.zeros(na*nb)
        ci0[0] = 1
    else:
        ci0 = ci0.ravel()

    #e, c = davidson.dsyev(hop, ci0, precond, tol=fci.tol, lindep=fci.lindep)
    e, c = fci.eig(hop, ci0, precond)
    return e, c.reshape(na,nb)


class FCISolver(object):
    def __init__(self, mol):
        self.mol = mol
        self.verbose = 0
        self.max_cycle = 50
        self.max_space = 12
        self.conv_threshold = 1e-8
        self.lindep = 1e-8
        self.max_memory = 1200 # MB
# level shift in precond
        self.level_shift = 1e-2

        self._keys = set(self.__dict__.keys() + ['_keys'])

    def dump_flags(self, verbose=None):
        if verbose is None:
            verbose = self.verbose
        log = pyscf.lib.logger.Logger(self.mol.stdout, verbose)
        log.info('******** CI flags ********')
        log.info('max. cycles = %d', self.max_cycle)
        log.info('conv_threshold = %g', self.conv_threshold)
        log.info('linear dependence = %g', self.lindep)
        log.info('level shift = %d', self.level_shift)
        log.info('max iter space = %d', self.max_space)
        log.info('max_memory %d MB', self.max_memory)


    def absorb_h1e(self, h1e, eri, norb, nelec, fac=1):
        return absorb_h1e(h1e, eri, norb, nelec, fac)

    def make_hdiag(self, h1e, eri, norb, nelec):
        return make_hdiag(h1e, eri, norb, nelec)

    def pspace(self, h1e, eri, norb, nelec, hdiag, np=400):
        return pspace(h1e, eri, norb, nelec, hdiag, np)

    def contract_1e(self, f1e, fcivec, norb, nelec, link_index=None, **kwargs):
        return contract_1e(f1e, fcivec, norb, nelec, link_index, **kwargs)

    def contract_2e(self, eri, fcivec, norb, nelec, link_index=None, **kwargs):
        return contract_2e(eri, fcivec, norb, nelec, link_index, **kwargs)

    def eig(self, op, x0, precond):
        return davidson.dsyev(op, x0, precond, self.conv_threshold,
                              self.max_cycle, self.max_space, self.lindep,
                              self.max_memory, verbose=self.verbose)

    def make_precond(self, hdiag, pspaceig, pspaceci, addr):
        def precond(r, e0, x0, *args):
            #h0e0 = h0 - numpy.eye(len(addr))*(e0-self.level_shift)
            h0e0inv = numpy.dot(pspaceci/(pspaceig-(e0-self.level_shift)),
                                pspaceci.T)
            hdiaginv = 1/(hdiag - (e0-self.level_shift))
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
        return precond
#    def make_precond(self, hdiag, *args):
#        return lambda x, e, *args: x/(hdiag-(e-self.level_shift))

    def kernel(self, h1e, eri, norb, nelec, ci0=None, **kwargs):
        self.mol.check_sanity(self)
        return kernel_ms1(self, h1e, eri, norb, nelec, ci0)

    def energy(self, h1e, eri, fcivec, norb, nelec, link_index=None):
        h2e = self.absorb_h1e(h1e, eri, norb, nelec, .5)
        ci1 = self.contract_2e(h2e, fcivec, norb, nelec, link_index)
        return numpy.dot(fcivec.reshape(-1), ci1.reshape(-1))

    def make_rdm1s(self, fcivec, norb, nelec, link_index=None, **kwargs):
        return make_rdm1s(fcivec, norb, nelec, link_index)

    def make_rdm1(self, fcivec, norb, nelec, link_index=None, **kwargs):
        return make_rdm1(fcivec, norb, nelec, link_index)

    def make_rdm12s(self, fcivec, norb, nelec, link_index=None, **kwargs):
        return make_rdm12s(fcivec, norb, nelec, link_index)

    def make_rdm12(self, fcivec, norb, nelec, link_index=None, **kwargs):
        return make_rdm12(fcivec, norb, nelec, link_index)

    def trans_rdm1s(self, cibra, ciket, norb, nelec, link_index=None, **kwargs):
        return trans_rdm1s(cibra, ciket, norb, nelec, link_index)

    def trans_rdm1(self, cibra, ciket, norb, nelec, link_index=None, **kwargs):
        return trans_rdm1(cibra, ciket, norb, nelec, link_index)

    def trans_rdm12s(self, cibra, ciket, norb, nelec, link_index=None, **kwargs):
        return trans_rdm12s(cibra, ciket, norb, nelec, link_index)

    def trans_rdm12(self, cibra, ciket, norb, nelec, link_index=None, **kwargs):
        return trans_rdm12(cibra, ciket, norb, nelec, link_index)


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
        #['H', ( 0.,-0.5   ,-1.   )],
        #['H', ( 0.,-0.5   ,-0.   )],
        ['H', ( 0.,-0.    ,-1.   )],
        ['H', ( 1.,-0.5   , 0.   )],
        ['H', ( 0., 1.    , 1.   )],
    ]

    mol.basis = {'H': 'sto-3g'}
    mol.build()

    m = scf.RHF(mol)
    ehf = m.scf()

    cis = FCISolver(mol)
    norb = m.mo_coeff.shape[1]
    nelec = mol.nelectron - 2
    h1e = reduce(numpy.dot, (m.mo_coeff.T, m.get_hcore(), m.mo_coeff))
    eri = ao2mo.incore.general(m._eri, (m.mo_coeff,)*4, compact=False)
    eri = eri.reshape(norb,norb,norb,norb)
    nea = nelec/2 + 1
    neb = nelec/2 - 1
    nelec = (nea, neb)

    e1 = cis.kernel(h1e, eri, norb, nelec)[0]
    print(e1, e1 - -7.7466756526056004)

