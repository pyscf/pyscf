#!/usr/bin/env python
# Copyright 2014-2023 The PySCF Developers. All Rights Reserved.
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
# Adapted for complex integral and general spin:
#         Huanchen Zhai <hczhai.ok@gmail.com>    Nov 16, 2022
#

import numpy
from pyscf import lib
from pyscf import ao2mo
from pyscf.fci import cistring
from pyscf.fci import direct_spin1
from pyscf.lib import logger


def contract_2e(eri, fcivec, norb, nelec, opt=None):
    '''Compute a^\\dagger_p a_q a^\\dagger_r a_s |CI>'''
    link_index = cistring.gen_linkstr_index(range(norb), nelec)
    na = cistring.num_strings(norb, nelec)
    t1 = numpy.zeros((norb, norb, na), dtype=eri.dtype)
    for str0, tab in enumerate(link_index):
        for a, i, str1, sign in tab:
            t1[a, i, str1] += sign * fcivec[str0]
    t1 = numpy.einsum('bjai,aiA->bjA', eri, t1, optimize=True)
    fcinew = numpy.zeros_like(fcivec)
    for str0, tab in enumerate(link_index):
        for a, i, str1, sign in tab:
            fcinew[str1] += sign * t1[a, i, str0]
    return fcinew


def absorb_h1e(h1e, eri, norb, nelec, fac=1):
    '''Modify 2e Hamiltonian to include 1e Hamiltonian contribution.'''
    f1e = h1e + 0.5 * numpy.einsum('jiik->jk', eri.transpose(0, 2, 3, 1), optimize=True)
    h2e = -1.0 * eri.transpose(0, 3, 2, 1)
    h2e[numpy.diag_indices(norb)] += f1e * (2.0 / (nelec + 1e-100))
    return h2e * fac


def make_hdiag(h1e, eri, norb, nelec, opt=None):
    occslist = cistring.gen_occslst(range(norb), nelec)
    diagjk = numpy.einsum('iijj->ij', eri.copy(), optimize=True)
    diagjk -= numpy.einsum('ijji->ij', eri, optimize=True)
    hdiag = []
    for occ in occslist:
        e1 = h1e[occ, occ].sum()
        e2 = diagjk[occ][:, occ].sum()
        hdiag.append(e1 + e2 * 0.5)
    return numpy.array(hdiag)


def kernel(h1e, eri, norb, nelec, ecore=0, nroots=1, verbose=3):
    h2e = absorb_h1e(h1e, eri, norb, nelec, 0.5)
    hop = lambda c: contract_2e(h2e, c, norb, nelec)
    hdiag = make_hdiag(h1e, eri, norb, nelec)
    precond = lambda x, e, *args: x / (hdiag - e + 1e-4)
    ci0 = get_init_guess(norb, nelec, nroots, hdiag)
    e, c = lib.davidson(hop, ci0, precond, verbose=verbose, nroots=nroots)
    return e + ecore, c


# dm_pq = <|p^+ q|>
def make_rdm1_slow(fcivec, norb, nelec, link_index=None):
    if link_index is None:
        link_index = cistring.gen_linkstr_index(range(norb), nelec)
    rdm1 = numpy.zeros((norb, norb), dtype=fcivec.dtype)
    for str0, tab in enumerate(link_index):
        for a, i, str1, sign in tab:
            rdm1[a, i] += sign * numpy.dot(fcivec[str1].conj(), fcivec[str0])
    return rdm1


# dm_pq = <|p^+ q|>
def make_rdm1(fcivec, norb, nelec, link_index=None):
    if nelec == 0:
        return numpy.zeros((norb, norb), dtype=fcivec.dtype)
    nd = cistring.num_strings(norb, nelec - 1)
    index_d = cistring.gen_des_str_index(range(norb), nelec)
    dci = numpy.zeros((nd, norb), dtype=fcivec.dtype)
    for str0, tab in enumerate(index_d):
        for _, i, str1, sign in tab:
            dci[str1, i] += sign * fcivec[str0]
    return dci.conj().T @ dci


# dm_pq,rs = <|p^+ q r^+ s|>
def make_rdm12(fcivec, norb, nelec, link_index=None, reorder=True):
    if link_index is None:
        link_index = cistring.gen_linkstr_index(range(norb), nelec)
    link_index = cistring.gen_linkstr_index(range(norb), nelec)
    rdm1 = numpy.zeros((norb, norb), dtype=fcivec.dtype)
    rdm2 = numpy.zeros((norb, norb, norb, norb), dtype=fcivec.dtype)
    for str0, tab in enumerate(link_index):
        t1 = numpy.zeros((norb, norb), dtype=fcivec.dtype)
        for a, i, str1, sign in tab:
            t1[i, a] += sign * fcivec[str1]
        rdm1 += fcivec[str0].conj() * t1
        # i^+ j|0> => <0|j^+ i, so swap i and j
        rdm2 += numpy.einsum('ij,kl->jikl', t1.conj(), t1, optimize=True)
    return reorder_rdm(rdm1, rdm2) if reorder else (rdm1, rdm2)


def reorder_rdm(rdm1, rdm2, inplace=True):
    nmo = rdm1.shape[0]
    rdm2 = (rdm2 if inplace else rdm2.copy()).reshape(nmo, nmo, nmo, nmo)
    rdm2[(slice(None), ) + numpy.diag_indices(nmo)] -= rdm1[:, None, :]
    rdm2 = -rdm2.transpose(0, 3, 2, 1)
    return rdm1, rdm2


def get_init_guess(norb, nelec, nroots, hdiag):
    '''Initial guess is the single Slater determinant
    '''
    na = cistring.num_strings(norb, nelec)
    ci0 = []
    if hdiag.size <= nroots:
        addrs = numpy.arange(hdiag.size)
    else:
        addrs = numpy.argpartition(hdiag, nroots-1)[:nroots]
    for addr in addrs:
        x = numpy.zeros((na, ), dtype=hdiag.dtype)
        x[addr] = 1
        ci0.append(x)

    # Add noise
    ci0[0][0 ] += 1e-5 + 1e-6j if ci0[0].dtype == complex else 1e-5
    ci0[0][-1] -= 1e-5 + 1e-6j if ci0[0].dtype == complex else 1e-5
    return ci0


def kernel_dhf(fci, h1e, eri, norb, nelec, ci0=None, link_index=None,
               tol=None, lindep=None, max_cycle=None, max_space=None,
               nroots=None, davidson_only=None, pspace_size=None,
               max_memory=None, verbose=None, ecore=0, **kwargs):
    if nroots is None: nroots = fci.nroots
    if davidson_only is None: davidson_only = fci.davidson_only
    if pspace_size is None: pspace_size = fci.pspace_size
    if max_memory is None:
        max_memory = fci.max_memory - lib.current_memory()[0]
    log = logger.new_logger(fci, verbose)
    link_index = cistring.gen_linkstr_index(range(norb), nelec)
    na = cistring.num_strings(norb, nelec)
    hdiag = fci.make_hdiag(h1e, eri, norb, nelec)
    nroots = min(hdiag.size, nroots)
    precond = lib.make_diag_precond(hdiag, fci.level_shift)

    h2e = fci.absorb_h1e(h1e, eri, norb, nelec, 0.5)
    def hop(c):
        return fci.contract_2e(h2e, c, norb, nelec, link_index)

    def init_guess():
        if callable(getattr(fci, 'get_init_guess', None)):
            return fci.get_init_guess(norb, nelec, nroots, hdiag)
        else:
            x0 = []
            for i in range(nroots):
                x = numpy.zeros(na, dtype=eri.dtype)
                x[i] = 1
                x0.append(x)
            return x0

    if ci0 is None:
        ci0 = init_guess  # lazy initialization to reduce memory footprint
    elif not callable(ci0):
        if len(ci0) < nroots:
            for i in range(len(ci0), nroots):
                x = numpy.zeros(na, dtype=eri.dtype)
                x[i] = 1
                ci0.append(x)

    if tol is None: tol = fci.conv_tol
    if lindep is None: lindep = fci.lindep
    if max_cycle is None: max_cycle = fci.max_cycle
    if max_space is None: max_space = fci.max_space
    tol_residual = getattr(fci, 'conv_tol_residual', None)

    with lib.with_omp_threads(fci.threads):
        e, c = fci.eig(hop, ci0, precond, tol=tol, lindep=lindep,
                       max_cycle=max_cycle, max_space=max_space, nroots=nroots,
                       max_memory=max_memory, verbose=log, follow_state=True,
                       tol_residual=tol_residual, **kwargs)
    if nroots > 1:
        return e + ecore, [ci.view(direct_spin1.FCIvector) for ci in c]
    else:
        return e + ecore, c.view(direct_spin1.FCIvector)


###############################################################
# ghf/dhf-integral direct-CI driver
###############################################################

class FCISolver(direct_spin1.FCISolver):

    def absorb_h1e(self, h1e, eri, norb, nelec, fac=1):
        return absorb_h1e(h1e, eri, norb, nelec, fac)

    def make_hdiag(self, h1e, eri, norb, nelec):
        return make_hdiag(h1e, eri, norb, nelec)

    def pspace(self, h1e, eri, norb, nelec, hdiag, np=400):
        return NotImplemented

    def contract_1e(self, f1e, fcivec, norb, nelec, link_index=None, **kwargs):
        return NotImplemented

    def contract_2e(self, eri, fcivec, norb, nelec, link_index=None, **kwargs):
        return contract_2e(eri, fcivec, norb, nelec, link_index, **kwargs)

    def spin_square(self, fcivec, norb, nelec):
        return NotImplemented

    def make_rdm1(self, fcivec, norb, nelec, link_index=None):
        return make_rdm1(fcivec, norb, nelec, link_index)

    def make_rdm12(self, fcivec, norb, nelec, link_index=None, reorder=True):
        return make_rdm12(fcivec, norb, nelec, link_index, reorder)

    def make_rdm2(self, fcivec, norb, nelec, link_index=None, reorder=True):
        return self.make_rdm12(fcivec, norb, nelec, link_index, reorder)[1]

    def energy(self, h1e, eri, fcivec, norb, nelec, link_index=None):
        h2e = self.absorb_h1e(h1e, eri, norb, nelec, 0.5)
        ci1 = self.contract_2e(h2e, fcivec, norb, nelec, link_index)
        return numpy.dot(fcivec.conj(), ci1)

    def get_init_guess(self, norb, nelec, nroots, hdiag):
        return get_init_guess(norb, nelec, nroots, hdiag)

    def kernel(self, h1e, eri, norb, nelec, ci0=None,
               tol=None, lindep=None, max_cycle=None, max_space=None,
               nroots=None, davidson_only=None, pspace_size=None,
               orbsym=None, wfnsym=None, ecore=0, **kwargs):
        self.norb = norb
        self.nelec = nelec
        self.eci, self.ci = \
                kernel_dhf(self, h1e, eri, norb, nelec, ci0, None,
                           tol, lindep, max_cycle, max_space, nroots,
                           davidson_only, pspace_size, ecore=ecore, **kwargs)
        return self.eci, self.ci

FCI = FCISolver

if __name__ == '__main__':
    from functools import reduce
    from pyscf import gto
    from pyscf import scf

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

    m = scf.GHF(mol).run(conv_tol=1e-14)

    norb = m.mo_coeff.shape[1]
    h1e = reduce(numpy.dot, (m.mo_coeff.conj().T, m.get_hcore(), m.mo_coeff))
    mo_a, mo_b = m.mo_coeff[:mol.nao], m.mo_coeff[mol.nao:]
    eri = ao2mo.restore(4, ao2mo.general(m._eri, (mo_a, mo_a, mo_b, mo_b)), norb)
    eri = eri + eri.transpose(1, 0)
    eri += ao2mo.restore(4, ao2mo.full(m._eri, mo_a), norb)
    eri += ao2mo.restore(4, ao2mo.full(m._eri, mo_b), norb)
    eri = ao2mo.restore(1, eri, norb)

    nelec = mol.nelectron - 2

    e1 = kernel(h1e, eri, norb, nelec)[0]
    print(e1, e1 - -7.9766331504361414)
