#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#
# FCI solver for Singlet state
#
# Other files in the directory
# direct_spin0 singlet
# direct_spin1 arbitary number of alpha and beta electrons, based on RHF/ROHF
#              MO integrals
# direct_uhf   arbitary number of alpha and beta electrons, based on UHF
#              MO integrals
#

import sys
import ctypes
import numpy
import scipy.linalg
import pyscf.lib
import pyscf.gto
import pyscf.ao2mo
from pyscf.fci import cistring
from pyscf.fci import rdm
from pyscf.fci import direct_spin1

libfci = pyscf.lib.load_library('libfci')

def contract_1e(f1e, fcivec, norb, nelec, link_index=None):
    assert(fcivec.flags.c_contiguous)
    if link_index is None:
        if isinstance(nelec, (int, numpy.integer)):
            neleca = nelec//2
        else:
            neleca, nelecb = nelec
            assert(neleca == nelecb)
        link_index = cistring.gen_linkstr_index_trilidx(range(norb), neleca)
    na,nlink,_ = link_index.shape
    ci1 = numpy.empty((na,na))
    f1e_tril = pyscf.lib.pack_tril(f1e)
    libfci.FCIcontract_1e_spin0(f1e_tril.ctypes.data_as(ctypes.c_void_p),
                                fcivec.ctypes.data_as(ctypes.c_void_p),
                                ci1.ctypes.data_as(ctypes.c_void_p),
                                ctypes.c_int(norb), ctypes.c_int(na),
                                ctypes.c_int(nlink),
                                link_index.ctypes.data_as(ctypes.c_void_p))
# no *.5 because FCIcontract_2e_spin0 only compute half of the contraction
    return pyscf.lib.transpose_sum(ci1, inplace=True)

# Note eri is NOT the 2e hamiltonian matrix, the 2e hamiltonian is
# h2e = eri_{pq,rs} p^+ q r^+ s
#     = (pq|rs) p^+ r^+ s q - (pq|rs) \delta_{qr} p^+ s
# so eri is defined as
#       eri_{pq,rs} = (pq|rs) - (1/Nelec) \sum_q (pq|qs)
# to restore the symmetry between pq and rs,
#       eri_{pq,rs} = (pq|rs) - (.5/Nelec) [\sum_q (pq|qs) + \sum_p (pq|rp)]
# Please refer to the treatment in direct_spin1.absorb_h1e
# the input fcivec should be symmetrized
def contract_2e(eri, fcivec, norb, nelec, link_index=None):
    assert(fcivec.flags.c_contiguous)
    eri = pyscf.ao2mo.restore(4, eri, norb)
    if link_index is None:
        if isinstance(nelec, (int, numpy.integer)):
            neleca = nelec//2
        else:
            neleca, nelecb = nelec
            assert(neleca == nelecb)
        link_index = cistring.gen_linkstr_index_trilidx(range(norb), neleca)
    na,nlink,_ = link_index.shape
    ci1 = numpy.empty((na,na))

    libfci.FCIcontract_2e_spin0(eri.ctypes.data_as(ctypes.c_void_p),
                                fcivec.ctypes.data_as(ctypes.c_void_p),
                                ci1.ctypes.data_as(ctypes.c_void_p),
                                ctypes.c_int(norb), ctypes.c_int(na),
                                ctypes.c_int(nlink),
                                link_index.ctypes.data_as(ctypes.c_void_p))
# no *.5 because FCIcontract_2e_spin0 only compute half of the contraction
    return pyscf.lib.transpose_sum(ci1, inplace=True)

def absorb_h1e(*args, **kwargs):
    return direct_spin1.absorb_h1e(*args, **kwargs)

def make_hdiag(h1e, eri, norb, nelec):
    if isinstance(nelec, (int, numpy.integer)):
        neleca = nelec//2
    else:
        neleca, nelecb = nelec
        assert(neleca == nelecb)
    h1e = numpy.ascontiguousarray(h1e)
    eri = pyscf.ao2mo.restore(1, eri, norb)
    link_index = cistring.gen_linkstr_index(range(norb), neleca)
    na = link_index.shape[0]
    occslist = numpy.asarray(link_index[:,:neleca,0], order='C')
    hdiag = numpy.empty((na,na))
    jdiag = numpy.asarray(numpy.einsum('iijj->ij',eri), order='C')
    kdiag = numpy.asarray(numpy.einsum('ijji->ij',eri), order='C')
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
    if isinstance(nelec, (int, numpy.integer)):
        neleca = nelec//2
    else:
        neleca, nelecb = nelec
        assert(neleca == nelecb)
    h1e = numpy.ascontiguousarray(h1e)
    eri = pyscf.ao2mo.restore(1, eri, norb)
    na = cistring.num_strings(norb, neleca)
    addr = numpy.argsort(hdiag)[:np]
# symmetrize addra/addrb
    addra = addr // na
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
    h0 = pyscf.lib.hermi_triu_(h0)
    return addr, h0

# be careful with single determinant initial guess. It may lead to the
# eigvalue of first davidson iter being equal to hdiag
def kernel(h1e, eri, norb, nelec, ci0=None, level_shift=.001, tol=1e-10,
           lindep=1e-14, max_cycle=50, nroots=1, **kwargs):
    cis = FCISolver(None)
    cis.level_shift = level_shift
    cis.conv_tol = tol
    cis.lindep = lindep
    cis.max_cycle = max_cycle
    cis.nroots = nroots

    unknown = []
    for k, v in kwargs.items():
        setattr(cis, k, v)
        if not hasattr(cis, k):
            unknown.append(k)
    if unknown:
        sys.stderr.write('Unknown keys %s for FCI kernel %s\n' %
                         (str(unknown), __name__))
    return kernel_ms0(cis, h1e, eri, norb, nelec, ci0=ci0)

# dm_pq = <|p^+ q|>
def make_rdm1(fcivec, norb, nelec, link_index=None):
    rdm1 = rdm.make_rdm1('FCImake_rdm1a', fcivec, fcivec,
                         norb, nelec, link_index)
    return rdm1 * 2

# alpha and beta 1pdm
def make_rdm1s(fcivec, norb, nelec, link_index=None):
    rdm1 = rdm.make_rdm1('FCImake_rdm1a', fcivec, fcivec,
                         norb, nelec, link_index)
    return (rdm1, rdm1)

# Chemist notation
def make_rdm12(fcivec, norb, nelec, link_index=None, reorder=True):
    #dm1, dm2 = rdm.make_rdm12('FCIrdm12kern_spin0', fcivec, fcivec,
    #                          norb, nelec, link_index, 1)
# NOT use FCIrdm12kern_spin0 because for small system, the kernel may call
# direct diagonalization, which may not fulfil  fcivec = fcivet.T
    dm1, dm2 = rdm.make_rdm12('FCIrdm12kern_sf', fcivec, fcivec,
                              norb, nelec, link_index, 1)
    if reorder:
        dm1, dm2 = rdm.reorder_rdm(dm1, dm2, True)
    return dm1, dm2

# dm_pq = <I|p^+ q|J>
def trans_rdm1s(cibra, ciket, norb, nelec, link_index=None):
    if link_index is None:
        if isinstance(nelec, (int, numpy.integer)):
            neleca = nelec//2
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
def trans_rdm12(cibra, ciket, norb, nelec, link_index=None, reorder=True):
    dm1, dm2 = rdm.make_rdm12('FCItdm12kern_sf', cibra, ciket,
                              norb, nelec, link_index, 2)
    if reorder:
        dm1, dm2 = rdm.reorder_rdm(dm1, dm2, True)
    return dm1, dm2

def energy(h1e, eri, fcivec, norb, nelec, link_index=None):
    h2e = direct_spin1.absorb_h1e(h1e, eri, norb, nelec, .5)
    ci1 = contract_2e(h2e, fcivec, norb, nelec, link_index)
    return numpy.dot(fcivec.ravel(), ci1.ravel())

def get_init_guess(norb, nelec, nroots, hdiag):
    if isinstance(nelec, (int, numpy.integer)):
        nelecb = nelec//2
        neleca = nelec - nelecb
    else:
        neleca, nelecb = nelec
    na = cistring.num_strings(norb, neleca)
    nb = cistring.num_strings(norb, nelecb)

    init_strs = []
    iroot = 0
    for addr in numpy.argsort(hdiag):
        addra = addr // nb
        addrb = addr % nb
        if (addrb,addra) not in init_strs:  # avoid initial guess linear dependency
            init_strs.append((addra,addrb))
            iroot += 1
            if iroot >= nroots:
                break
    ci0 = []
    for addra,addrb in init_strs:
        x = numpy.zeros((na,nb))
        if addra == addrb == 0:
            x[addra,addrb] = 1
        else:
            x[addra,addrb] = x[addrb,addra] = numpy.sqrt(.5)
        ci0.append(x.ravel())
    return ci0


###############################################################
# direct-CI driver
###############################################################

def kernel_ms0(fci, h1e, eri, norb, nelec, ci0=None, **kwargs):
    if isinstance(nelec, (int, numpy.integer)):
        neleca = nelec//2
    else:
        neleca, nelecb = nelec
        assert(neleca == nelecb)
    h1e = numpy.ascontiguousarray(h1e)
    eri = numpy.ascontiguousarray(eri)
    link_index = cistring.gen_linkstr_index_trilidx(range(norb), neleca)
    na = link_index.shape[0]
    hdiag = fci.make_hdiag(h1e, eri, norb, nelec)

    addr, h0 = fci.pspace(h1e, eri, norb, nelec, hdiag, fci.pspace_size)
    pw, pv = scipy.linalg.eigh(h0)
# The degenerated wfn can break symmetry.  The davidson iteration with proper
# initial guess doesn't have this issue
    if not fci.davidson_only:
        if len(addr) == na*na:
            if na*na == 1:
                return pw[0], pv[:,0]
            elif fci.nroots > 1:
                civec = numpy.empty((fci.nroots,na*na))
                civec[:,addr] = pv[:,:fci.nroots].T
                civec = civec.reshape(fci.nroots,na,na)
                try:
                    return pw[:fci.nroots], [_check_(ci) for ci in civec]
                except ValueError:
                    pass
            elif abs(pw[0]-pw[1]) > 1e-12:
                civec = numpy.empty((na*na))
                civec[addr] = pv[:,0]
                civec = civec.reshape(na,na)
                civec = pyscf.lib.transpose_sum(civec) * .5
                # direct diagonalization may lead to triplet ground state
##TODO: optimize initial guess.  Using pspace vector as initial guess may have
## spin problems.  The 'ground state' of psapce vector may have different spin
## state to the true ground state.
                try:
                    return pw[0], _check_(civec.reshape(na,na))
                except ValueError:
                    pass

    precond = fci.make_precond(hdiag, pw, pv, addr)

    h2e = fci.absorb_h1e(h1e, eri, norb, nelec, .5)
    def hop(c):
        hc = fci.contract_2e(h2e, c, norb, nelec, link_index)
        return hc.ravel()

#TODO: check spin of initial guess
    if ci0 is None:
        if hasattr(fci, 'get_init_guess'):
            ci0 = fci.get_init_guess(norb, nelec, fci.nroots, hdiag)
        else:
            ci0 = []
            for i in range(fci.nroots):
                x = numpy.zeros(na,na)
                if addr[i] == 0:
                    x[0,0] = 1
                else:
                    addra = addr[i] // na
                    addrb = addr[i] % na
                    x[addra,addrb] = x[addrb,addra] = numpy.sqrt(.5)
                ci0.append(x.ravel())
    else:
        if isinstance(ci0, numpy.ndarray) and ci0.size == na*na:
            ci0 = [ci0.ravel()]
        else:
            ci0 = [x.ravel() for x in ci0]

    #e, c = pyscf.lib.davidson(hop, ci0, precond, tol=fci.conv_tol, lindep=fci.lindep)
    e, c = fci.eig(hop, ci0, precond, **kwargs)
    if fci.nroots > 1:
        return e, [_check_(ci.reshape(na,na)) for ci in c]
    else:
        return e, _check_(c.reshape(na,na))

def _check_(c):
    c = pyscf.lib.transpose_sum(c)
    c *= .5
    if abs(numpy.linalg.norm(c)-1) > 1e-9:
        raise ValueError('State not singlet %g' % abs(numpy.linalg.norm(c)-1))
    return c


class FCISolver(direct_spin1.FCISolver):

    def absorb_h1e(self, h1e, eri, norb, nelec, fac=1):
        return direct_spin1.absorb_h1e(h1e, eri, norb, nelec, fac)

    def make_hdiag(self, h1e, eri, norb, nelec):
        return make_hdiag(h1e, eri, norb, nelec)

    def pspace(self, h1e, eri, norb, nelec, hdiag, np=400):
        return pspace(h1e, eri, norb, nelec, hdiag, np)

    def contract_1e(self, f1e, fcivec, norb, nelec, link_index=None, **kwargs):
        return contract_1e(f1e, fcivec, norb, nelec, link_index, **kwargs)

    def contract_2e(self, eri, fcivec, norb, nelec, link_index=None, **kwargs):
        return contract_2e(eri, fcivec, norb, nelec, link_index, **kwargs)

    def eig(self, op, x0, precond, **kwargs):
        opts = {'tol': self.conv_tol,
                'max_cycle': self.max_cycle,
                'max_space': self.max_space,
                'lindep': self.lindep,
                'max_memory': self.max_memory,
                'nroots': self.nroots,
                'verbose': pyscf.lib.logger.Logger(self.stdout, self.verbose)}
        if self.nroots == 1 and x0[0].size > 6.5e7: # 500MB
            opts['lessio'] = True
        opts.update(kwargs)
        return pyscf.lib.davidson(op, x0, precond, **opts)

    def make_precond(self, hdiag, pspaceig, pspaceci, addr):
        return direct_spin1.make_pspace_precond(hdiag, pspaceig, pspaceci, addr,
                                                self.level_shift)

    def get_init_guess(self, norb, nelec, nroots, hdiag):
        return get_init_guess(norb, nelec, nroots, hdiag)

    def kernel(self, h1e, eri, norb, nelec, ci0=None, **kwargs):
        if self.verbose > pyscf.lib.logger.QUIET:
            pyscf.gto.mole.check_sanity(self, self._keys, self.stdout)
        e, ci = kernel_ms0(self, h1e, eri, norb, nelec, ci0, **kwargs)
        return e, ci

    def energy(self, h1e, eri, fcivec, norb, nelec, link_index=None):
        h2e = self.absorb_h1e(h1e, eri, norb, nelec, .5)
        ci1 = self.contract_2e(h2e, fcivec, norb, nelec, link_index)
        return numpy.dot(fcivec.reshape(-1), ci1.reshape(-1))

    def make_rdm1s(self, fcivec, norb, nelec, link_index=None):
        return make_rdm1s(fcivec, norb, nelec, link_index)

    def make_rdm1(self, fcivec, norb, nelec, link_index=None):
        return make_rdm1(fcivec, norb, nelec, link_index)

    def make_rdm12(self, fcivec, norb, nelec, link_index=None, reorder=True):
        return make_rdm12(fcivec, norb, nelec, link_index, reorder)

    def make_rdm2(self, fcivec, norb, nelec, link_index=None, reorder=True):
        return make_rdm12(fcivec, norb, nelec, link_index, reorder)[1]

    def trans_rdm1s(self, cibra, ciket, norb, nelec, link_index=None):
        return trans_rdm1s(cibra, ciket, norb, nelec, link_index)

    def trans_rdm1(self, cibra, ciket, norb, nelec, link_index=None):
        return trans_rdm1(cibra, ciket, norb, nelec, link_index)

    def trans_rdm12(self, cibra, ciket, norb, nelec, link_index=None,
                    reorder=True):
        return trans_rdm12(cibra, ciket, norb, nelec, link_index, reorder)



if __name__ == '__main__':
    import time
    from functools import reduce
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

    cis = FCISolver(mol)
    norb = m.mo_coeff.shape[1]
    nelec = mol.nelectron
    h1e = reduce(numpy.dot, (m.mo_coeff.T, m.get_hcore(), m.mo_coeff))
    eri = ao2mo.incore.general(m._eri, (m.mo_coeff,)*4, compact=False)
    e, c = cis.kernel(h1e, eri, norb, nelec)
    print(e - -15.9977886375)
    print('t',time.clock())

