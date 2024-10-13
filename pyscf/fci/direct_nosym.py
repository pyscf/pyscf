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

import warnings
import ctypes
import numpy
import scipy.linalg
from pyscf import lib
from pyscf import ao2mo
from pyscf.fci import cistring
from pyscf.fci import direct_spin1

libfci = direct_spin1.libfci

def contract_1e(h1e, fcivec, norb, nelec, link_index=None):
    h1e = numpy.asarray(h1e, order='C')
    fcivec = numpy.asarray(fcivec, order='C')
    link_indexa, link_indexb = _unpack(norb, nelec, link_index)

    na, nlinka = link_indexa.shape[:2]
    nb, nlinkb = link_indexb.shape[:2]
    assert fcivec.size == na*nb
    assert fcivec.dtype == h1e.dtype == numpy.float64
    ci1 = numpy.zeros_like(fcivec)

    libfci.FCIcontract_a_1e_nosym(h1e.ctypes.data_as(ctypes.c_void_p),
                                  fcivec.ctypes.data_as(ctypes.c_void_p),
                                  ci1.ctypes.data_as(ctypes.c_void_p),
                                  ctypes.c_int(norb),
                                  ctypes.c_int(na), ctypes.c_int(nb),
                                  ctypes.c_int(nlinka), ctypes.c_int(nlinkb),
                                  link_indexa.ctypes.data_as(ctypes.c_void_p),
                                  link_indexb.ctypes.data_as(ctypes.c_void_p))
    libfci.FCIcontract_b_1e_nosym(h1e.ctypes.data_as(ctypes.c_void_p),
                                  fcivec.ctypes.data_as(ctypes.c_void_p),
                                  ci1.ctypes.data_as(ctypes.c_void_p),
                                  ctypes.c_int(norb),
                                  ctypes.c_int(na), ctypes.c_int(nb),
                                  ctypes.c_int(nlinka), ctypes.c_int(nlinkb),
                                  link_indexa.ctypes.data_as(ctypes.c_void_p),
                                  link_indexb.ctypes.data_as(ctypes.c_void_p))
    return ci1.view(direct_spin1.FCIvector)

def contract_2e(eri, fcivec, norb, nelec, link_index=None):
    r'''Contract the 2-electron Hamiltonian with a FCI vector to get a new FCI
    vector.

    Note the input arg eri is NOT the 2e hamiltonian matrix, the 2e hamiltonian is

    .. math::

        h2e &= eri_{pq,rs} p^+ q r^+ s \\
            &= (pq|rs) p^+ r^+ s q - (pq|rs) \delta_{qr} p^+ s

    So eri is defined as

    .. math::

        eri_{pq,rs} = (pq|rs) - (1/Nelec) \sum_q (pq|qs)

    to restore the symmetry between pq and rs,

    .. math::

        eri_{pq,rs} = (pq|rs) - (.5/Nelec) [\sum_q (pq|qs) + \sum_p (pq|rp)]

    See also :func:`direct_nosym.absorb_h1e`
    '''
    link_indexa, link_indexb = _unpack(norb, nelec, link_index)
    na, nlinka = link_indexa.shape[:2]
    nb, nlinkb = link_indexb.shape[:2]
    assert fcivec.size == na*nb
    if fcivec.dtype == eri.dtype == numpy.float64:
        fcivec = numpy.asarray(fcivec, order='C')
        eri = numpy.asarray(eri, order='C')
        ci1 = numpy.empty_like(fcivec)
        libfci.FCIcontract_2es1(eri.ctypes.data_as(ctypes.c_void_p),
                                fcivec.ctypes.data_as(ctypes.c_void_p),
                                ci1.ctypes.data_as(ctypes.c_void_p),
                                ctypes.c_int(norb),
                                ctypes.c_int(na), ctypes.c_int(nb),
                                ctypes.c_int(nlinka), ctypes.c_int(nlinkb),
                                link_indexa.ctypes.data_as(ctypes.c_void_p),
                                link_indexb.ctypes.data_as(ctypes.c_void_p))
        return ci1.view(direct_spin1.FCIvector)

    ciR = numpy.asarray(fcivec.real, order='C')
    ciI = numpy.asarray(fcivec.imag, order='C')
    eriR = numpy.asarray(eri.real, order='C')
    eriI = numpy.asarray(eri.imag, order='C')
    link_index = (link_indexa, link_indexb)
    outR  = contract_2e(eriR, ciR, norb, nelec, link_index=link_index)
    outR -= contract_2e(eriI, ciI, norb, nelec, link_index=link_index)
    outI  = contract_2e(eriR, ciI, norb, nelec, link_index=link_index)
    outI += contract_2e(eriI, ciR, norb, nelec, link_index=link_index)
    out = outR.astype(numpy.complex128)
    out.imag = outI
    return out

def absorb_h1e(h1e, eri, norb, nelec, fac=1):
    '''Modify 2e Hamiltonian to include 1e Hamiltonian contribution.
    '''
    if not isinstance(nelec, (int, numpy.number)):
        nelec = sum(nelec)
    if h1e.dtype == eri.dtype == numpy.float64:
        h2e = ao2mo.restore(1, eri.copy(), norb)
    else:
        assert eri.ndim == 4
        h2e = eri.astype(dtype=numpy.result_type(h1e, eri), copy=True)
    f1e = h1e - numpy.einsum('jiik->jk', h2e) * .5
    f1e = f1e * (1./(nelec+1e-100))
    for k in range(norb):
        h2e[k,k,:,:] += f1e
        h2e[:,:,k,k] += f1e
    return h2e * fac

def energy(h1e, eri, fcivec, norb, nelec, link_index=None):
    '''Compute the FCI electronic energy for given Hamiltonian and FCI vector.
    '''
    h2e = absorb_h1e(h1e, eri, norb, nelec, .5)
    ci1 = contract_2e(h2e, fcivec, norb, nelec, link_index)
    return numpy.dot(fcivec.reshape(-1), ci1.reshape(-1))

def make_hdiag(h1e, eri, norb, nelec, compress=False):
    if h1e.dtype == numpy.complex128:
        h1e = h1e.real.copy()
    if eri.dtype == numpy.complex128:
        eri = eri.real.copy()
    return direct_spin1.make_hdiag(h1e, eri, norb, nelec, compress)


class FCISolver(direct_spin1.FCISolver):
    def __init__(self, *args, **kwargs):
        direct_spin1.FCISolver.__init__(self, *args, **kwargs)
        # pspace constructor only supports Hermitian Hamiltonian
        self.davidson_only = True

    def contract_1e(self, h1e, fcivec, norb, nelec, link_index=None):
        return contract_1e(h1e, fcivec, norb, nelec, link_index)

    def contract_2e(self, eri, fcivec, norb, nelec, link_index=None):
        return contract_2e(eri, fcivec, norb, nelec, link_index)

    def absorb_h1e(self, h1e, eri, norb, nelec, fac=1):
        return absorb_h1e(h1e, eri, norb, nelec, fac)

    def make_hdiag(self, h1e, eri, norb, nelec, compress=False):
        nelec = direct_spin1._unpack_nelec(nelec, self.spin)
        return make_hdiag(h1e, eri, norb, nelec, compress)

    def kernel(self, h1e, eri, norb, nelec, ci0=None,
               tol=None, lindep=None, max_cycle=None, max_space=None,
               nroots=None, davidson_only=None, pspace_size=None,
               orbsym=None, wfnsym=None, ecore=0, **kwargs):
        if isinstance(nelec, (int, numpy.number)):
            nelecb = nelec//2
            neleca = nelec - nelecb
        else:
            neleca, nelecb = nelec
        davidson_only = True
        link_indexa = cistring.gen_linkstr_index(range(norb), neleca)
        link_indexb = cistring.gen_linkstr_index(range(norb), nelecb)
        e, c = direct_spin1.kernel_ms1(self, h1e, eri, norb, nelec, ci0,
                                       (link_indexa,link_indexb),
                                       tol, lindep, max_cycle, max_space, nroots,
                                       davidson_only, pspace_size, ecore=ecore,
                                       **kwargs)
        self.eci, self.ci = e, c
        return e, c

    def eig(self, op, x0=None, precond=None, **kwargs):
        if isinstance(op, numpy.ndarray):
            self.converged = True
            return scipy.linalg.eigh(op)

        # TODO: check the hermitian of Hamiltonian then determine whether to
        # call the non-hermitian diagonalization solver davidson_nosym1

        warnings.warn('direct_nosym.kernel is not able to diagonalize '
                      'non-Hermitian Hamiltonian. If h1e and h2e is not '
                      'hermtian, calling symmetric diagonalization in eig '
                      'can lead to wrong results.')

        self.converged, e, ci = \
                lib.davidson1(lambda xs: [op(x) for x in xs],
                              x0, precond, lessio=self.lessio, **kwargs)
        if kwargs.get('nroots', 1) == 1:
            self.converged = self.converged[0]
            e = e[0]
            ci = ci[0]
        return e, ci

FCI = FCISolver

def _unpack(norb, nelec, link_index):
    if link_index is None:
        if isinstance(nelec, (int, numpy.number)):
            nelecb = nelec//2
            neleca = nelec - nelecb
        else:
            neleca, nelecb = nelec
        link_indexa = cistring.gen_linkstr_index(range(norb), neleca)
        link_indexb = cistring.gen_linkstr_index(range(norb), nelecb)
        return link_indexa, link_indexb
    else:
        return link_index
