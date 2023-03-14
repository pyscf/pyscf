#!/usr/bin/env python
# Copyright 2014-2019 The PySCF Developers. All Rights Reserved.
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

import sys
import ctypes
import numpy
from pyscf import lib
from pyscf import ao2mo
from pyscf.lib import logger
from pyscf import symm
from pyscf.scf.hf_symm import map_degeneracy
from pyscf.fci import cistring
from pyscf.fci import direct_spin0
from pyscf.fci import direct_spin1
from pyscf.fci import direct_spin1_symm
from pyscf.fci import addons
from pyscf.fci.spin_op import contract_ss
from pyscf import __config__

libfci = lib.load_library('libfci')

TOTIRREPS = 8

def contract_1e(f1e, fcivec, norb, nelec, link_index=None, orbsym=None):
    return direct_spin0.contract_1e(f1e, fcivec, norb, nelec, link_index)

# Note eri is NOT the 2e hamiltonian matrix, the 2e hamiltonian is
# h2e = eri_{pq,rs} p^+ q r^+ s
#     = (pq|rs) p^+ r^+ s q - (pq|rs) \delta_{qr} p^+ s
# so eri is defined as
#       eri_{pq,rs} = (pq|rs) - (1/Nelec) \sum_q (pq|qs)
# to restore the symmetry between pq and rs,
#       eri_{pq,rs} = (pq|rs) - (.5/Nelec) [\sum_q (pq|qs) + \sum_p (pq|rp)]
# Please refer to the treatment in direct_spin1.absorb_h1e
# the input fcivec should be symmetrized
def contract_2e(eri, fcivec, norb, nelec, link_index=None, orbsym=None, wfnsym=0):
    if orbsym is None:
        return direct_spin0.contract_2e(eri, fcivec, norb, nelec, link_index)

    eri = ao2mo.restore(4, eri, norb)
    neleca, nelecb = direct_spin1._unpack_nelec(nelec)
    assert (neleca == nelecb)
    link_indexa = direct_spin0._unpack(norb, nelec, link_index)
    na, nlinka = link_indexa.shape[:2]
    eri_irs, rank_eri, irrep_eri = direct_spin1_symm.reorder_eri(eri, norb, orbsym)

    strsa = numpy.asarray(cistring.gen_strings4orblist(range(norb), neleca))
    aidx, link_indexa = direct_spin1_symm.gen_str_irrep(strsa, orbsym, link_indexa,
                                                        rank_eri, irrep_eri)

    Tirrep = ctypes.c_void_p*TOTIRREPS
    linka_ptr = Tirrep(*[x.ctypes.data_as(ctypes.c_void_p) for x in link_indexa])
    eri_ptrs = Tirrep(*[x.ctypes.data_as(ctypes.c_void_p) for x in eri_irs])
    dimirrep = (ctypes.c_int*TOTIRREPS)(*[x.shape[0] for x in eri_irs])
    fcivec_shape = fcivec.shape
    fcivec = fcivec.reshape((na,na), order='C')
    ci1new = numpy.zeros_like(fcivec)
    nas = (ctypes.c_int*TOTIRREPS)(*[x.size for x in aidx])

    ci0 = []
    ci1 = []
    wfnsym_in_d2h = wfnsym % 10
    for ir in range(TOTIRREPS):
        ma, mb = aidx[ir].size, aidx[wfnsym_in_d2h ^ ir].size
        ci0.append(numpy.zeros((ma,mb)))
        ci1.append(numpy.zeros((ma,mb)))
        if ma > 0 and mb > 0:
            lib.take_2d(fcivec, aidx[ir], aidx[wfnsym_in_d2h ^ ir], out=ci0[ir])
    ci0_ptrs = Tirrep(*[x.ctypes.data_as(ctypes.c_void_p) for x in ci0])
    ci1_ptrs = Tirrep(*[x.ctypes.data_as(ctypes.c_void_p) for x in ci1])
    libfci.FCIcontract_2e_symm1(eri_ptrs, ci0_ptrs, ci1_ptrs,
                                ctypes.c_int(norb), nas, nas,
                                ctypes.c_int(nlinka), ctypes.c_int(nlinka),
                                linka_ptr, linka_ptr, dimirrep,
                                ctypes.c_int(wfnsym_in_d2h))
    for ir in range(TOTIRREPS):
        if ci0[ir].size > 0:
            lib.takebak_2d(ci1new, ci1[ir], aidx[ir], aidx[wfnsym_in_d2h ^ ir])
    ci1 = lib.transpose_sum(ci1new, inplace=True).reshape(fcivec_shape)
    return ci1.view(direct_spin1.FCIvector)


def kernel(h1e, eri, norb, nelec, ci0=None, level_shift=1e-3, tol=1e-10,
           lindep=1e-14, max_cycle=50, max_space=12, nroots=1,
           davidson_only=False, pspace_size=400, orbsym=None, wfnsym=None,
           ecore=0, **kwargs):
    assert (len(orbsym) == norb)
    cis = FCISolver(None)
    cis.level_shift = level_shift
    cis.conv_tol = tol
    cis.lindep = lindep
    cis.max_cycle = max_cycle
    cis.max_space = max_space
    cis.nroots = nroots
    cis.davidson_only = davidson_only
    cis.pspace_size = pspace_size
    cis.orbsym = orbsym
    cis.wfnsym = wfnsym

    unknown = {}
    for k, v in kwargs.items():
        if not hasattr(cis, k):
            unknown[k] = v
        setattr(cis, k, v)
    if unknown:
        sys.stderr.write('Unknown keys %s for FCI kernel %s\n' %
                         (str(unknown.keys()), __name__))

    wfnsym = direct_spin1_symm._id_wfnsym(cis, norb, nelec, cis.orbsym,
                                          cis.wfnsym)
    if cis.wfnsym is not None and ci0 is None:
        ci0 = addons.symm_initguess(norb, nelec, orbsym, wfnsym)

    e, c = cis.kernel(h1e, eri, norb, nelec, ci0, ecore=ecore, **unknown)
    return e, c

make_rdm1 = direct_spin0.make_rdm1
make_rdm1s = direct_spin0.make_rdm1s
make_rdm12 = direct_spin0.make_rdm12

trans_rdm1s = direct_spin0.trans_rdm1s
trans_rdm1 = direct_spin0.trans_rdm1
trans_rdm12 = direct_spin0.trans_rdm12

def energy(h1e, eri, fcivec, norb, nelec, link_index=None, orbsym=None, wfnsym=0):
    h2e = direct_spin1.absorb_h1e(h1e, eri, norb, nelec) * .5
    ci1 = contract_2e(h2e, fcivec, norb, nelec, link_index, orbsym, wfnsym)
    return numpy.dot(fcivec.ravel(), ci1.ravel())

def get_init_guess(norb, nelec, nroots, hdiag, orbsym, wfnsym=0):
    neleca, nelecb = direct_spin1._unpack_nelec(nelec)
    assert (neleca == nelecb)
    strsa = cistring.gen_strings4orblist(range(norb), neleca)
    airreps = direct_spin1_symm._gen_strs_irrep(strsa, orbsym)
    na = nb = len(airreps)
    hdiag = hdiag.reshape(na,nb)

    ci0 = []
    iroot = 0
    sym_allowed = (airreps[:,None] ^ airreps) == wfnsym
    idx = numpy.arange(na)
    sym_allowed[idx[:,None] < idx] = False
    idx_a, idx_b = numpy.where(sym_allowed)
    for k in hdiag[idx_a,idx_b].argsort():
        addra, addrb = idx_a[k], idx_b[k]
        x = numpy.zeros((na, nb))
        if addra == addrb:
            x[addra,addrb] = 1
        else:
            x[addra,addrb] = x[addrb,addra] = numpy.sqrt(.5)
        ci0.append(x.ravel().view(direct_spin1.FCIvector))
        iroot += 1
        if iroot >= nroots:
            break

    if len(ci0) == 0:
        raise RuntimeError(f'Initial guess for symmetry {wfnsym} not found')
    return ci0

def get_init_guess_cyl_sym(norb, nelec, nroots, hdiag, orbsym, wfnsym=0):
    neleca, nelecb = direct_spin1._unpack_nelec(nelec)
    strsa = cistring.gen_strings4orblist(range(norb), neleca)
    airreps_d2h = direct_spin1_symm._gen_strs_irrep(strsa, orbsym)
    a_ls = direct_spin1_symm._strs_angular_momentum(strsa, orbsym)

    wfnsym_in_d2h = wfnsym % 10
    wfn_momentum = symm.basis.linearmole_irrep2momentum(wfnsym)
    na = nb = len(strsa)
    hdiag = hdiag.reshape(na,nb)
    degen = orbsym.degen_mapping
    ci0 = []
    iroot = 0
    wfn_ungerade = wfnsym_in_d2h >= 4
    a_ungerade = airreps_d2h >= 4
    sym_allowed = a_ungerade[:,None] == a_ungerade ^ wfn_ungerade
    # total angular momentum == wfn_momentum
    sym_allowed &= a_ls[:,None] == wfn_momentum - a_ls
    idx = numpy.arange(na)
    sym_allowed[idx[:,None] < idx] = False

    idx_a, idx_b = numpy.where(sym_allowed)
    for k in hdiag[idx_a,idx_b].argsort():
        addra, addrb = idx_a[k], idx_b[k]
        ca = direct_spin1_symm._cyl_sym_csf2civec(strsa, addra, orbsym, degen)
        cb = direct_spin1_symm._cyl_sym_csf2civec(strsa, addrb, orbsym, degen)
        if wfn_momentum > 0 or wfnsym in (0, 5):
            x = ca.real[:,None] * cb.real
            x-= ca.imag[:,None] * cb.imag
        else:
            x = ca.imag[:,None] * cb.real
            x+= ca.real[:,None] * cb.imag
        if addra == addrb:
            norm = numpy.linalg.norm(x)
        else:
            x = x + x.T
            norm = numpy.linalg.norm(x)
            if norm < 1e-7:
                continue
        x *= 1./norm
        ci0.append(x.ravel().view(direct_spin1.FCIvector))
        iroot += 1
        if iroot >= nroots:
            break

    if len(ci0) == 0:
        raise RuntimeError(f'Initial guess for symmetry {wfnsym} not found')
    return ci0


class FCISolver(direct_spin0.FCISolver):

    davidson_only = getattr(__config__, 'fci_direct_spin1_symm_FCI_davidson_only', True)

    # pspace may break point group symmetry
    pspace_size = getattr(__config__, 'fci_direct_spin1_symm_FCI_pspace_size', 0)

    def __init__(self, mol=None, **kwargs):
        direct_spin0.FCISolver.__init__(self, mol, **kwargs)
        # wfnsym will be guessed based on initial guess if it is None
        self.wfnsym = None

    def dump_flags(self, verbose=None):
        direct_spin0.FCISolver.dump_flags(self, verbose)
        log = logger.new_logger(self, verbose)
        if isinstance(self.wfnsym, str):
            log.info('specified CI wfn symmetry = %s', self.wfnsym)
        elif isinstance(self.wfnsym, (int, numpy.number)):
            groupname = getattr(self.mol, 'groupname', None)
            log.info('specified CI wfn symmetry = %s',
                     symm.irrep_id2name(groupname, self.wfnsym))

    def absorb_h1e(self, h1e, eri, norb, nelec, fac=1):
        return direct_spin1.absorb_h1e(h1e, eri, norb, nelec, fac)

    def make_hdiag(self, h1e, eri, norb, nelec):
        return direct_spin0.make_hdiag(h1e, eri, norb, nelec)

    def pspace(self, h1e, eri, norb, nelec, hdiag, np=400):
        return direct_spin0.pspace(h1e, eri, norb, nelec, hdiag, np)

    def contract_1e(self, f1e, fcivec, norb, nelec, link_index=None, **kwargs):
        return contract_1e(f1e, fcivec, norb, nelec, link_index, **kwargs)

    def contract_2e(self, eri, fcivec, norb, nelec, link_index=None,
                    orbsym=None, wfnsym=None, **kwargs):
        if orbsym is None: orbsym = self.orbsym
        if wfnsym is None: wfnsym = self.wfnsym
        wfnsym = direct_spin1_symm._id_wfnsym(self, norb, nelec, orbsym,
                                              wfnsym)
        return contract_2e(eri, fcivec, norb, nelec, link_index, orbsym, wfnsym, **kwargs)

    def get_init_guess(self, norb, nelec, nroots, hdiag):
        wfnsym = direct_spin1_symm._id_wfnsym(self, norb, nelec, self.orbsym,
                                              self.wfnsym)
        if getattr(self.mol, 'groupname', None) in ('Dooh', 'Coov'):
            return get_init_guess_cyl_sym(
                norb, nelec, nroots, hdiag, self.orbsym, wfnsym)
        else:
            return get_init_guess(norb, nelec, nroots, hdiag, self.orbsym, wfnsym)

    guess_wfnsym = direct_spin1_symm.guess_wfnsym

    def kernel(self, h1e, eri, norb, nelec, ci0=None,
               tol=None, lindep=None, max_cycle=None, max_space=None,
               nroots=None, davidson_only=None, pspace_size=None,
               orbsym=None, wfnsym=None, ecore=0, **kwargs):
        if nroots is None: nroots = self.nroots
        if orbsym is None: orbsym = self.orbsym
        if wfnsym is None: wfnsym = self.wfnsym
        if self.verbose >= logger.WARN:
            self.check_sanity()
        self.norb = norb
        self.nelec = nelec

        if (not hasattr(orbsym, 'degen_mapping') and
            getattr(self.mol, 'groupname', None) in ('Dooh', 'Coov')):
            degen_mapping = map_degeneracy(h1e.diagonal(), orbsym)
            orbsym = lib.tag_array(orbsym, degen_mapping=degen_mapping)

        wfnsym = self.guess_wfnsym(norb, nelec, ci0, orbsym, wfnsym, **kwargs)

        if wfnsym > 7:
            # Symmetry broken for Dooh and Coov groups is often observed.
            # A larger max_space is helpful to reduce the error. Also it is
            # hard to converge to high precision.
            if max_space is None and self.max_space == FCISolver.max_space:
                max_space = 20 + 7 * nroots
            if tol is None and self.conv_tol == FCISolver.conv_tol:
                tol = 1e-7

        with lib.temporary_env(self, orbsym=orbsym, wfnsym=wfnsym):
            e, c = direct_spin0.kernel_ms0(self, h1e, eri, norb, nelec, ci0, None,
                                           tol, lindep, max_cycle, max_space,
                                           nroots, davidson_only, pspace_size,
                                           ecore=ecore, **kwargs)
        self.eci, self.ci = e, c
        return e, c

FCI = FCISolver
