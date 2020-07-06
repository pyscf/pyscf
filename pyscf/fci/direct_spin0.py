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
FCI solver for Singlet state

Different FCI solvers are implemented to support different type of symmetry.
                    Symmetry
File                Point group   Spin singlet   Real hermitian*    Alpha/beta degeneracy
direct_spin0_symm   Yes           Yes            Yes                Yes
direct_spin1_symm   Yes           No             Yes                Yes
direct_spin0        No            Yes            Yes                Yes
direct_spin1        No            No             Yes                Yes
direct_uhf          No            No             Yes                No
direct_nosym        No            No             No**               Yes

*  Real hermitian Hamiltonian implies (ij|kl) = (ji|kl) = (ij|lk) = (ji|lk)
** Hamiltonian is real but not hermitian, (ij|kl) != (ji|kl) ...

direct_spin0 solver is specified for singlet state. However, calling this
solver sometimes ends up with the error "State not singlet x.xxxxxxe-06" due
to numerical issues. Calling direct_spin1 for singlet state is slightly
slower but more robust than direct_spin0 especially when combining to energy
penalty method (:func:`fix_spin_`)
'''

import ctypes
import numpy
from pyscf import lib
from pyscf import ao2mo
from pyscf.lib import logger
from pyscf.fci import cistring
from pyscf.fci import rdm
from pyscf.fci import direct_spin1
from pyscf.fci.spin_op import contract_ss

libfci = lib.load_library('libfci')

@lib.with_doc(direct_spin1.contract_1e.__doc__)
def contract_1e(f1e, fcivec, norb, nelec, link_index=None):
    fcivec = numpy.asarray(fcivec, order='C')
    link_index = _unpack(norb, nelec, link_index)
    na, nlink = link_index.shape[:2]
    assert(fcivec.size == na**2)
    ci1 = numpy.empty_like(fcivec)
    f1e_tril = lib.pack_tril(f1e)
    libfci.FCIcontract_1e_spin0(f1e_tril.ctypes.data_as(ctypes.c_void_p),
                                fcivec.ctypes.data_as(ctypes.c_void_p),
                                ci1.ctypes.data_as(ctypes.c_void_p),
                                ctypes.c_int(norb), ctypes.c_int(na),
                                ctypes.c_int(nlink),
                                link_index.ctypes.data_as(ctypes.c_void_p))
# no *.5 because FCIcontract_2e_spin0 only compute half of the contraction
    return lib.transpose_sum(ci1, inplace=True).reshape(fcivec.shape)

# Note eri is NOT the 2e hamiltonian matrix, the 2e hamiltonian is
# h2e = eri_{pq,rs} p^+ q r^+ s
#     = (pq|rs) p^+ r^+ s q - (pq|rs) \delta_{qr} p^+ s
# so eri is defined as
#       eri_{pq,rs} = (pq|rs) - (1/Nelec) \sum_q (pq|qs)
# to restore the symmetry between pq and rs,
#       eri_{pq,rs} = (pq|rs) - (.5/Nelec) [\sum_q (pq|qs) + \sum_p (pq|rp)]
# Please refer to the treatment in direct_spin1.absorb_h1e
# the input fcivec should be symmetrized
@lib.with_doc(direct_spin1.contract_2e.__doc__)
def contract_2e(eri, fcivec, norb, nelec, link_index=None):
    fcivec = numpy.asarray(fcivec, order='C')
    eri = ao2mo.restore(4, eri, norb)
    lib.transpose_sum(eri, inplace=True)
    eri *= .5
    link_index = _unpack(norb, nelec, link_index)
    na, nlink = link_index.shape[:2]
    assert(fcivec.size == na**2)
    ci1 = numpy.empty((na,na))

    libfci.FCIcontract_2e_spin0(eri.ctypes.data_as(ctypes.c_void_p),
                                fcivec.ctypes.data_as(ctypes.c_void_p),
                                ci1.ctypes.data_as(ctypes.c_void_p),
                                ctypes.c_int(norb), ctypes.c_int(na),
                                ctypes.c_int(nlink),
                                link_index.ctypes.data_as(ctypes.c_void_p))
# no *.5 because FCIcontract_2e_spin0 only compute half of the contraction
    return lib.transpose_sum(ci1, inplace=True).reshape(fcivec.shape)

absorb_h1e = direct_spin1.absorb_h1e

@lib.with_doc(direct_spin1.make_hdiag.__doc__)
def make_hdiag(h1e, eri, norb, nelec):
    hdiag = direct_spin1.make_hdiag(h1e, eri, norb, nelec)
    na = int(numpy.sqrt(hdiag.size))
# symmetrize hdiag to reduce numerical error
    hdiag = lib.transpose_sum(hdiag.reshape(na,na), inplace=True) * .5
    return hdiag.ravel()

pspace = direct_spin1.pspace

# be careful with single determinant initial guess. It may lead to the
# eigvalue of first davidson iter being equal to hdiag
def kernel(h1e, eri, norb, nelec, ci0=None, level_shift=1e-3, tol=1e-10,
           lindep=1e-14, max_cycle=50, max_space=12, nroots=1,
           davidson_only=False, pspace_size=400, orbsym=None, wfnsym=None,
           ecore=0, **kwargs):
    e, c = direct_spin1._kfactory(FCISolver, h1e, eri, norb, nelec, ci0, level_shift,
                                  tol, lindep, max_cycle, max_space, nroots,
                                  davidson_only, pspace_size, ecore=ecore, **kwargs)
    return e, c

# dm[p,q] = <|q^+ p|>
@lib.with_doc(direct_spin1.make_rdm1.__doc__)
def make_rdm1(fcivec, norb, nelec, link_index=None):
    rdm1 = rdm.make_rdm1('FCImake_rdm1a', fcivec, fcivec,
                         norb, nelec, link_index)
    return rdm1 * 2

# alpha and beta 1pdm
@lib.with_doc(direct_spin1.make_rdm1s.__doc__)
def make_rdm1s(fcivec, norb, nelec, link_index=None):
    rdm1 = rdm.make_rdm1('FCImake_rdm1a', fcivec, fcivec,
                         norb, nelec, link_index)
    return rdm1, rdm1

# Chemist notation
@lib.with_doc(direct_spin1.make_rdm12.__doc__)
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

# dm[p,q] = <I|q^+ p|J>
@lib.with_doc(direct_spin1.trans_rdm1s.__doc__)
def trans_rdm1s(cibra, ciket, norb, nelec, link_index=None):
    if link_index is None:
        if isinstance(nelec, (int, numpy.number)):
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

@lib.with_doc(direct_spin1.trans_rdm1.__doc__)
def trans_rdm1(cibra, ciket, norb, nelec, link_index=None):
    rdm1a, rdm1b = trans_rdm1s(cibra, ciket, norb, nelec, link_index)
    return rdm1a + rdm1b

# dm[p,q,r,s] = <I|p^+ q r^+ s|J>
@lib.with_doc(direct_spin1.trans_rdm12.__doc__)
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
    if isinstance(nelec, (int, numpy.number)):
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
        if addra == addrb:
            x[addra,addrb] = 1
        else:
            x[addra,addrb] = x[addrb,addra] = numpy.sqrt(.5)
        ci0.append(x.ravel())

    # Add noise
    ci0[0][0 ] += 1e-5
    ci0[0][-1] -= 1e-5
    return ci0


###############################################################
# direct-CI driver
###############################################################

def kernel_ms0(fci, h1e, eri, norb, nelec, ci0=None, link_index=None,
               tol=None, lindep=None, max_cycle=None, max_space=None,
               nroots=None, davidson_only=None, pspace_size=None,
               max_memory=None, verbose=None, ecore=0, **kwargs):
    if nroots is None: nroots = fci.nroots
    if davidson_only is None: davidson_only = fci.davidson_only
    if pspace_size is None: pspace_size = fci.pspace_size
    if max_memory is None:
        max_memory = fci.max_memory - lib.current_memory()[0]
    log = logger.new_logger(fci, verbose)

    assert(fci.spin is None or fci.spin == 0)
    assert(0 <= numpy.sum(nelec) <= norb*2)

    link_index = _unpack(norb, nelec, link_index)
    h1e = numpy.ascontiguousarray(h1e)
    eri = numpy.ascontiguousarray(eri)
    na = link_index.shape[0]

    if max_memory < na**2*6*8e-6:
        log.warn('Not enough memory for FCI solver. '
                 'The minimal requirement is %.0f MB', na**2*60e-6)

    hdiag = fci.make_hdiag(h1e, eri, norb, nelec)
    nroots = min(hdiag.size, nroots)

    try:
        addr, h0 = fci.pspace(h1e, eri, norb, nelec, hdiag, max(pspace_size,nroots))
        if pspace_size > 0:
            pw, pv = fci.eig(h0)
        else:
            pw = pv = None

        if pspace_size >= na*na and ci0 is None and not davidson_only:
# The degenerated wfn can break symmetry.  The davidson iteration with proper
# initial guess doesn't have this issue
            if na*na == 1:
                return pw[0]+ecore, pv[:,0].reshape(1,1)
            elif nroots > 1:
                civec = numpy.empty((nroots,na*na))
                civec[:,addr] = pv[:,:nroots].T
                civec = civec.reshape(nroots,na,na)
                try:
                    return pw[:nroots]+ecore, [_check_(ci) for ci in civec]
                except ValueError:
                    pass
            elif abs(pw[0]-pw[1]) > 1e-12:
                civec = numpy.empty((na*na))
                civec[addr] = pv[:,0]
                civec = civec.reshape(na,na)
                civec = lib.transpose_sum(civec) * .5
                # direct diagonalization may lead to triplet ground state
##TODO: optimize initial guess.  Using pspace vector as initial guess may have
## spin problems.  The 'ground state' of psapce vector may have different spin
## state to the true ground state.
                try:
                    return pw[0]+ecore, _check_(civec.reshape(na,na))
                except ValueError:
                    pass
    except NotImplementedError:
        addr = [0]
        pw = pv = None

    precond = fci.make_precond(hdiag, pw, pv, addr)

    h2e = fci.absorb_h1e(h1e, eri, norb, nelec, .5)
    def hop(c):
        hc = fci.contract_2e(h2e, c.reshape(na,na), norb, nelec, link_index)
        return hc.ravel()

#TODO: check spin of initial guess
    if ci0 is None:
        if callable(getattr(fci, 'get_init_guess', None)):
            ci0 = lambda: fci.get_init_guess(norb, nelec, nroots, hdiag)
        else:
            def ci0():
                x0 = []
                for i in range(nroots):
                    x = numpy.zeros((na,na))
                    addra = addr[i] // na
                    addrb = addr[i] % na
                    if addra == addrb:
                        x[addra,addrb] = 1
                    else:
                        x[addra,addrb] = x[addrb,addra] = numpy.sqrt(.5)
                    x0.append(x.ravel())
                return x0
    elif not callable(ci0):
        if isinstance(ci0, numpy.ndarray) and ci0.size == na*na:
            ci0 = [ci0.ravel()]
        else:
            ci0 = [x.ravel() for x in ci0]

    if tol is None: tol = fci.conv_tol
    if lindep is None: lindep = fci.lindep
    if max_cycle is None: max_cycle = fci.max_cycle
    if max_space is None: max_space = fci.max_space
    tol_residual = getattr(fci, 'conv_tol_residual', None)

    with lib.with_omp_threads(fci.threads):
        #e, c = lib.davidson(hop, ci0, precond, tol=fci.conv_tol, lindep=fci.lindep)
        e, c = fci.eig(hop, ci0, precond, tol=tol, lindep=lindep,
                       max_cycle=max_cycle, max_space=max_space, nroots=nroots,
                       max_memory=max_memory, verbose=log, follow_state=True,
                       tol_residual=tol_residual, **kwargs)
    if nroots > 1:
        return e+ecore, [_check_(ci.reshape(na,na)) for ci in c]
    else:
        return e+ecore, _check_(c.reshape(na,na))

def _check_(c):
    c = lib.transpose_sum(c, inplace=True)
    c *= .5
    norm = numpy.linalg.norm(c)
    if abs(norm-1) > 1e-6:
        raise ValueError('State not singlet %g' % abs(numpy.linalg.norm(c)-1))
    return c/norm


class FCISolver(direct_spin1.FCISolver):

    def make_hdiag(self, h1e, eri, norb, nelec):
        return make_hdiag(h1e, eri, norb, nelec)

    def contract_1e(self, f1e, fcivec, norb, nelec, link_index=None, **kwargs):
        return contract_1e(f1e, fcivec, norb, nelec, link_index, **kwargs)

    def contract_2e(self, eri, fcivec, norb, nelec, link_index=None, **kwargs):
        return contract_2e(eri, fcivec, norb, nelec, link_index, **kwargs)

    def get_init_guess(self, norb, nelec, nroots, hdiag):
        return get_init_guess(norb, nelec, nroots, hdiag)

    def kernel(self, h1e, eri, norb, nelec, ci0=None,
               tol=None, lindep=None, max_cycle=None, max_space=None,
               nroots=None, davidson_only=None, pspace_size=None,
               orbsym=None, wfnsym=None, ecore=0, **kwargs):
        if self.verbose >= logger.WARN:
            self.check_sanity()
        self.norb = norb
        self.nelec = nelec
        self.eci, self.ci = \
                kernel_ms0(self, h1e, eri, norb, nelec, ci0, None,
                           tol, lindep, max_cycle, max_space, nroots,
                           davidson_only, pspace_size, ecore=ecore, **kwargs)
        return self.eci, self.ci

    def energy(self, h1e, eri, fcivec, norb, nelec, link_index=None):
        h2e = self.absorb_h1e(h1e, eri, norb, nelec, .5)
        ci1 = self.contract_2e(h2e, fcivec, norb, nelec, link_index)
        return numpy.dot(fcivec.reshape(-1), ci1.reshape(-1))

    def make_rdm1s(self, fcivec, norb, nelec, link_index=None):
        return make_rdm1s(fcivec, norb, nelec, link_index)

    def make_rdm1(self, fcivec, norb, nelec, link_index=None):
        return make_rdm1(fcivec, norb, nelec, link_index)

    @lib.with_doc(make_rdm12.__doc__)
    def make_rdm12(self, fcivec, norb, nelec, link_index=None, reorder=True):
        return make_rdm12(fcivec, norb, nelec, link_index, reorder)

    def trans_rdm1s(self, cibra, ciket, norb, nelec, link_index=None):
        return trans_rdm1s(cibra, ciket, norb, nelec, link_index)

    def trans_rdm1(self, cibra, ciket, norb, nelec, link_index=None):
        return trans_rdm1(cibra, ciket, norb, nelec, link_index)

    @lib.with_doc(trans_rdm12.__doc__)
    def trans_rdm12(self, cibra, ciket, norb, nelec, link_index=None,
                    reorder=True):
        return trans_rdm12(cibra, ciket, norb, nelec, link_index, reorder)

    def gen_linkstr(self, norb, nelec, tril=True, spin=None):
        if isinstance(nelec, (int, numpy.number)):
            neleca = nelec//2
        else:
            neleca, nelecb = nelec
            assert(neleca == nelecb)
        if tril:
            link_index = cistring.gen_linkstr_index_trilidx(range(norb), neleca)
        else:
            link_index = cistring.gen_linkstr_index(range(norb), neleca)
        return link_index

FCI = FCISolver


def _unpack(norb, nelec, link_index):
    if link_index is None:
        if isinstance(nelec, (int, numpy.number)):
            neleca = nelec//2
        else:
            neleca, nelecb = nelec
            assert(neleca == nelecb)
        return cistring.gen_linkstr_index_trilidx(range(norb), neleca)
    else:
        return link_index


if __name__ == '__main__':
    import time
    from functools import reduce
    from pyscf import gto
    from pyscf import scf

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

