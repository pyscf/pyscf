#!/usr/bin/env python
# Copyright 2014-2021 The PySCF Developers. All Rights Reserved.
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
'''

import ctypes
import numpy
from pyscf import lib
from pyscf import ao2mo
from pyscf.fci import cistring
from pyscf.fci import direct_spin1
from pyscf.fci.spin_op import spin_square

libfci = direct_spin1.libfci

# When the spin-orbitals do not have the degeneracy on spatial part,
# there is only one version of FCI which is close to _spin1 solver.
# The inputs: h1e has two parts (h1e_a, h1e_b),
# h2e has three parts (h2e_aa, h2e_ab, h2e_bb)

def contract_1e(f1e, fcivec, norb, nelec, link_index=None):
    fcivec = numpy.asarray(fcivec, order='C')
    link_indexa, link_indexb = direct_spin1._unpack(norb, nelec, link_index)
    na, nlinka = link_indexa.shape[:2]
    nb, nlinkb = link_indexb.shape[:2]
    assert (fcivec.size == na*nb)
    ci1 = numpy.zeros_like(fcivec)
    f1e_tril = lib.pack_tril(f1e[0])
    libfci.FCIcontract_a_1e(f1e_tril.ctypes.data_as(ctypes.c_void_p),
                            fcivec.ctypes.data_as(ctypes.c_void_p),
                            ci1.ctypes.data_as(ctypes.c_void_p),
                            ctypes.c_int(norb),
                            ctypes.c_int(na), ctypes.c_int(nb),
                            ctypes.c_int(nlinka), ctypes.c_int(nlinkb),
                            link_indexa.ctypes.data_as(ctypes.c_void_p),
                            link_indexb.ctypes.data_as(ctypes.c_void_p))
    f1e_tril = lib.pack_tril(f1e[1])
    libfci.FCIcontract_b_1e(f1e_tril.ctypes.data_as(ctypes.c_void_p),
                            fcivec.ctypes.data_as(ctypes.c_void_p),
                            ci1.ctypes.data_as(ctypes.c_void_p),
                            ctypes.c_int(norb),
                            ctypes.c_int(na), ctypes.c_int(nb),
                            ctypes.c_int(nlinka), ctypes.c_int(nlinkb),
                            link_indexa.ctypes.data_as(ctypes.c_void_p),
                            link_indexb.ctypes.data_as(ctypes.c_void_p))
    return ci1.view(direct_spin1.FCIvector)

def contract_2e(eri, fcivec, norb, nelec, link_index=None):
    fcivec = numpy.asarray(fcivec, order='C')
    g2e_aa = ao2mo.restore(4, eri[0], norb)
    g2e_ab = ao2mo.restore(4, eri[1], norb)
    g2e_bb = ao2mo.restore(4, eri[2], norb)

    link_indexa, link_indexb = direct_spin1._unpack(norb, nelec, link_index)
    na, nlinka = link_indexa.shape[:2]
    nb, nlinkb = link_indexb.shape[:2]
    assert (fcivec.size == na*nb)
    ci1 = numpy.empty_like(fcivec)

    libfci.FCIcontract_uhf2e(g2e_aa.ctypes.data_as(ctypes.c_void_p),
                             g2e_ab.ctypes.data_as(ctypes.c_void_p),
                             g2e_bb.ctypes.data_as(ctypes.c_void_p),
                             fcivec.ctypes.data_as(ctypes.c_void_p),
                             ci1.ctypes.data_as(ctypes.c_void_p),
                             ctypes.c_int(norb),
                             ctypes.c_int(na), ctypes.c_int(nb),
                             ctypes.c_int(nlinka), ctypes.c_int(nlinkb),
                             link_indexa.ctypes.data_as(ctypes.c_void_p),
                             link_indexb.ctypes.data_as(ctypes.c_void_p))
    return ci1.view(direct_spin1.FCIvector)

def contract_2e_hubbard(u, fcivec, norb, nelec, opt=None):
    neleca, nelecb = direct_spin1._unpack_nelec(nelec)
    u_aa, u_ab, u_bb = u

    strsa = numpy.asarray(cistring.gen_strings4orblist(range(norb), neleca))
    strsb = numpy.asarray(cistring.gen_strings4orblist(range(norb), nelecb))
    na = cistring.num_strings(norb, neleca)
    nb = cistring.num_strings(norb, nelecb)
    fcivec = fcivec.reshape(na,nb)
    fcinew = numpy.zeros_like(fcivec)

    if u_aa != 0:  # u * n_alpha^+ n_alpha
        for i in range(norb):
            maska = (strsa & (1 << i)) > 0
            fcinew[maska] += u_aa * fcivec[maska]
    if u_ab != 0:  # u * (n_alpha^+ n_beta + n_beta^+ n_alpha)
        for i in range(norb):
            maska = (strsa & (1 << i)) > 0
            maskb = (strsb & (1 << i)) > 0
            fcinew[maska[:,None] & maskb] += 2*u_ab * fcivec[maska[:,None] & maskb]
    if u_bb != 0:  # u * n_beta^+ n_beta
        for i in range(norb):
            maskb = (strsb & (1 << i)) > 0
            fcinew[:,maskb] += u_bb * fcivec[:,maskb]
    return fcinew.view(direct_spin1.FCIvector)

def make_hdiag(h1e, eri, norb, nelec, compress=False):
    neleca, nelecb = direct_spin1._unpack_nelec(nelec)
    h1e_a = numpy.ascontiguousarray(h1e[0])
    h1e_b = numpy.ascontiguousarray(h1e[1])
    g2e_aa = ao2mo.restore(1, eri[0], norb)
    g2e_ab = ao2mo.restore(1, eri[1], norb)
    g2e_bb = ao2mo.restore(1, eri[2], norb)

    occslsta = occslstb = cistring.gen_occslst(range(norb), neleca)
    if neleca != nelecb:
        occslstb = cistring.gen_occslst(range(norb), nelecb)
    na = len(occslsta)
    nb = len(occslstb)

    hdiag = numpy.empty(na*nb)
    jdiag_aa = numpy.asarray(numpy.einsum('iijj->ij',g2e_aa), order='C')
    jdiag_ab = numpy.asarray(numpy.einsum('iijj->ij',g2e_ab), order='C')
    jdiag_bb = numpy.asarray(numpy.einsum('iijj->ij',g2e_bb), order='C')
    kdiag_aa = numpy.asarray(numpy.einsum('ijji->ij',g2e_aa), order='C')
    kdiag_bb = numpy.asarray(numpy.einsum('ijji->ij',g2e_bb), order='C')
    libfci.FCImake_hdiag_uhf(hdiag.ctypes.data_as(ctypes.c_void_p),
                             h1e_a.ctypes.data_as(ctypes.c_void_p),
                             h1e_b.ctypes.data_as(ctypes.c_void_p),
                             jdiag_aa.ctypes.data_as(ctypes.c_void_p),
                             jdiag_ab.ctypes.data_as(ctypes.c_void_p),
                             jdiag_bb.ctypes.data_as(ctypes.c_void_p),
                             kdiag_aa.ctypes.data_as(ctypes.c_void_p),
                             kdiag_bb.ctypes.data_as(ctypes.c_void_p),
                             ctypes.c_int(norb),
                             ctypes.c_int(na), ctypes.c_int(nb),
                             ctypes.c_int(neleca), ctypes.c_int(nelecb),
                             occslsta.ctypes.data_as(ctypes.c_void_p),
                             occslstb.ctypes.data_as(ctypes.c_void_p))
    return numpy.asarray(hdiag)

def absorb_h1e(h1e, eri, norb, nelec, fac=1):
    if not isinstance(nelec, (int, numpy.number)):
        nelec = sum(nelec)
    h1e_a, h1e_b = h1e
    h2e_aa = ao2mo.restore(1, eri[0], norb).copy()
    h2e_ab = ao2mo.restore(1, eri[1], norb).copy()
    h2e_bb = ao2mo.restore(1, eri[2], norb).copy()
    f1e_a = h1e_a - numpy.einsum('jiik->jk', h2e_aa) * .5
    f1e_b = h1e_b - numpy.einsum('jiik->jk', h2e_bb) * .5
    f1e_a *= 1./(nelec+1e-100)
    f1e_b *= 1./(nelec+1e-100)
    for k in range(norb):
        h2e_aa[:,:,k,k] += f1e_a
        h2e_aa[k,k,:,:] += f1e_a
        h2e_ab[:,:,k,k] += f1e_a
        h2e_ab[k,k,:,:] += f1e_b
        h2e_bb[:,:,k,k] += f1e_b
        h2e_bb[k,k,:,:] += f1e_b
    return (ao2mo.restore(4, h2e_aa, norb) * fac,
            ao2mo.restore(4, h2e_ab, norb) * fac,
            ao2mo.restore(4, h2e_bb, norb) * fac)

def pspace(h1e, eri, norb, nelec, hdiag=None, np=400):
    if norb >= 64:
        raise NotImplementedError('norb >= 64')

    if h1e[0].dtype == numpy.complex128 or eri[0].dtype == numpy.complex128:
        raise NotImplementedError('Complex Hamiltonian')

    neleca, nelecb = direct_spin1._unpack_nelec(nelec)
    h1e_a = numpy.ascontiguousarray(h1e[0])
    h1e_b = numpy.ascontiguousarray(h1e[1])
    g2e_aa = ao2mo.restore(1, eri[0], norb)
    g2e_ab = ao2mo.restore(1, eri[1], norb)
    g2e_bb = ao2mo.restore(1, eri[2], norb)
    if hdiag is None:
        hdiag = make_hdiag(h1e, eri, norb, nelec, compress=False)
    na = cistring.num_strings(norb, neleca)
    nb = cistring.num_strings(norb, nelecb)
    assert hdiag.size == na * nb
    if hdiag.size <= np:
        addr = numpy.arange(hdiag.size)
    else:
        try:
            addr = numpy.argpartition(hdiag, np-1)[:np]
        except AttributeError:
            addr = numpy.argsort(hdiag)[:np]
    addra = addr // nb
    addrb = addr % nb
    stra = cistring.addrs2str(norb, neleca, addra)
    strb = cistring.addrs2str(norb, nelecb, addrb)
    np = len(addr)
    h0 = numpy.zeros((np,np))
    libfci.FCIpspace_h0tril_uhf(h0.ctypes.data_as(ctypes.c_void_p),
                                h1e_a.ctypes.data_as(ctypes.c_void_p),
                                h1e_b.ctypes.data_as(ctypes.c_void_p),
                                g2e_aa.ctypes.data_as(ctypes.c_void_p),
                                g2e_ab.ctypes.data_as(ctypes.c_void_p),
                                g2e_bb.ctypes.data_as(ctypes.c_void_p),
                                stra.ctypes.data_as(ctypes.c_void_p),
                                strb.ctypes.data_as(ctypes.c_void_p),
                                ctypes.c_int(norb), ctypes.c_int(np))

    for i in range(np):
        h0[i,i] = hdiag[addr[i]]
    h0 = lib.hermi_triu(h0)
    return addr, h0


# be careful with single determinant initial guess. It may lead to the
# eigvalue of first davidson iter being equal to hdiag
def kernel(h1e, eri, norb, nelec, ci0=None, level_shift=1e-3, tol=1e-10,
           lindep=1e-14, max_cycle=50, max_space=12, nroots=1,
           davidson_only=False, pspace_size=400, orbsym=None, wfnsym=None,
           ecore=0, **kwargs):
    return direct_spin1._kfactory(FCISolver, h1e, eri, norb, nelec, ci0, level_shift,
                                  tol, lindep, max_cycle, max_space, nroots,
                                  davidson_only, pspace_size, ecore=ecore, **kwargs)

def energy(h1e, eri, fcivec, norb, nelec, link_index=None):
    h2e = absorb_h1e(h1e, eri, norb, nelec, .5)
    ci1 = contract_2e(h2e, fcivec, norb, nelec, link_index)
    return numpy.dot(fcivec.reshape(-1), ci1.reshape(-1))

# dm_pq = <|p^+ q|>
make_rdm1s = direct_spin1.make_rdm1s

# spatial part of DM, dm_pq = <|p^+ q|>
def make_rdm1(fcivec, norb, nelec, link_index=None):
    raise ValueError('Spin trace for UHF-FCI density matrices.')

make_rdm12s = direct_spin1.make_rdm12s
trans_rdm1s = direct_spin1.trans_rdm1s

# spatial part of DM
def trans_rdm1(cibra, ciket, norb, nelec, link_index=None):
    raise ValueError('Spin trace for UHF-FCI density matrices.')

trans_rdm12s = direct_spin1.trans_rdm12s


###############################################################
# uhf-integral direct-CI driver
###############################################################

class FCISolver(direct_spin1.FCISolver):

    absorb_h1e = staticmethod(absorb_h1e)
    make_hdiag = staticmethod(make_hdiag)
    pspace = staticmethod(pspace)

    contract_1e = lib.module_method(contract_1e)
    contract_2e = lib.module_method(contract_2e)
    make_rdm1 = lib.module_method(make_rdm1)
    trans_rdm1 = lib.module_method(trans_rdm1)
    spin_square = lib.module_method(spin_square)

FCI = FCISolver
