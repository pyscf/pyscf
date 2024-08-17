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
from pyscf import __config__

libfci = direct_spin1.libfci

def contract_2e(eri, fcivec, norb, nelec, link_index=None, orbsym=None, wfnsym=0):
    link_index = direct_spin1._unpack(norb, nelec, link_index)
    if isinstance(link_index, numpy.ndarray):
        # For backward compatibility
        link_index = (link_index, link_index)
    return direct_spin1_symm.contract_2e(eri, fcivec, norb, nelec, link_index,
                                         orbsym, wfnsym)

energy = direct_spin1_symm.energy
kernel = direct_spin1_symm.kernel

make_rdm1 = direct_spin0.make_rdm1
make_rdm1s = direct_spin0.make_rdm1s
make_rdm12 = direct_spin0.make_rdm12

trans_rdm1s = direct_spin0.trans_rdm1s
trans_rdm1 = direct_spin0.trans_rdm1
trans_rdm12 = direct_spin0.trans_rdm12

def get_init_guess(norb, nelec, nroots, hdiag, orbsym, wfnsym=0):
    neleca, nelecb = direct_spin1._unpack_nelec(nelec)
    assert (neleca == nelecb)

    strsa = cistring.gen_strings4orblist(range(norb), neleca)
    na = len(strsa)
    airreps = direct_spin1_symm._gen_strs_irrep(strsa, orbsym)

    sym_allowed = (airreps[:,None] ^ airreps) == wfnsym
    idx = numpy.arange(na)
    sym_allowed[idx[:,None] < idx] = False
    idx_a, idx_b = numpy.where(sym_allowed)

    hdiag = hdiag.reshape(na,na)[idx_a,idx_b]
    if hdiag.size <= nroots:
        hdiag_indices = numpy.arange(hdiag.size)
    else:
        hdiag_indices = numpy.argpartition(hdiag, nroots-1)[:nroots]

    ci0 = []
    for k in hdiag_indices:
        addra, addrb = idx_a[k], idx_b[k]
        x = numpy.zeros((na, na))
        if addra == addrb:
            x[addra,addrb] = 1
        else:
            x[addra,addrb] = x[addrb,addra] = numpy.sqrt(.5)
        ci0.append(x.ravel().view(direct_spin1.FCIvector))

    if len(ci0) == 0:
        raise lib.exceptions.WfnSymmetryError(
            f'Initial guess for symmetry {wfnsym} not found')
    return ci0

def get_init_guess_cyl_sym(norb, nelec, nroots, hdiag, orbsym, wfnsym=0):
    neleca, nelecb = direct_spin1._unpack_nelec(nelec)
    na = cistring.num_strings(norb, neleca)
    ci0_guess = direct_spin1_symm.get_init_guess_cyl_sym(
            norb, nelec, nroots, hdiag, orbsym, wfnsym)
    ci0 = []
    for x in ci0_guess:
        x = x.reshape(na, na)
        x = x + x.T
        norm = numpy.linalg.norm(x)
        if norm < 1e-3:
            continue
        x *= 1./norm
        ci0.append(x.ravel().view(direct_spin1.FCIvector))

    if len(ci0) == 0:
        raise lib.exceptions.WfnSymmetryError(
            f'Initial guess for symmetry {wfnsym} not found')
    return ci0


class FCISolver(direct_spin0.FCISolver):

    _keys = {'wfnsym', 'sym_allowed_idx'}

    davidson_only = getattr(__config__, 'fci_direct_spin1_symm_FCI_davidson_only', True)
    pspace_size = getattr(__config__, 'fci_direct_spin1_symm_FCI_pspace_size', 400)

    def __init__(self, mol=None, **kwargs):
        # wfnsym will be guessed based on initial guess if it is None
        self.wfnsym = None
        self.sym_allowed_idx = None
        direct_spin0.FCISolver.__init__(self, mol, **kwargs)

    absorb_h1e = direct_spin1.FCISolver.absorb_h1e

    dump_flags = direct_spin1_symm.FCISolver.dump_flags
    make_hdiag = direct_spin1_symm.FCISolver.make_hdiag
    pspace = direct_spin1_symm.FCISolver.pspace
    contract_1e = direct_spin1_symm.FCISolver.contract_1e
    contract_ss = direct_spin1_symm.FCISolver.contract_ss
    guess_wfnsym = direct_spin1_symm.guess_wfnsym
    kernel = direct_spin1_symm.FCISolver.kernel

    def contract_2e(self, eri, fcivec, norb, nelec, link_index=None,
                    orbsym=None, wfnsym=None, **kwargs):
        if orbsym is None: orbsym = self.orbsym
        if wfnsym is None: wfnsym = self.wfnsym
        wfnsym = direct_spin1_symm._id_wfnsym(self, norb, nelec, orbsym, wfnsym)
        nelec = direct_spin1._unpack_nelec(nelec, self.spin)
        civec = contract_2e(eri, fcivec, norb, nelec, link_index, orbsym, wfnsym)
        na = cistring.num_strings(norb, nelec[0])
        if civec.size != na**2:
            s_idx = numpy.hstack(self.sym_allowed_idx)
            x, y = divmod(s_idx, na)
            ci1 = numpy.empty(na**2)
            ci1[y*na+x] = civec
            civec += ci1[s_idx]
            civec *= .5
        else:
            civec = lib.transpose_sum(civec.reshape(na,na), inplace=True)
            civec *= .5
        return civec

    def get_init_guess(self, norb, nelec, nroots, hdiag, orbsym=None, wfnsym=None):
        if orbsym is None: orbsym = self.orbsym
        if wfnsym is None:
            wfnsym = direct_spin1_symm._id_wfnsym(
                self, norb, nelec, orbsym, self.wfnsym)
        s_idx = numpy.hstack(self.sym_allowed_idx)
        if getattr(self.mol, 'groupname', None) in ('Dooh', 'Coov'):
            ci0 = get_init_guess_cyl_sym(
                norb, nelec, nroots, hdiag, orbsym, wfnsym)
        else:
            nelec = direct_spin1._unpack_nelec(nelec, self.spin)
            na = cistring.num_strings(norb, nelec[0])
            if hdiag.size != na * na:
                hdiag, hdiag0 = numpy.empty(na*na), hdiag
                hdiag[:] = 1e9
                hdiag[numpy.hstack(self.sym_allowed_idx)] = hdiag0
            ci0 = get_init_guess(norb, nelec, nroots, hdiag.ravel(),
                                 orbsym, wfnsym)
        return [x[s_idx] for x in ci0]

FCI = FCISolver
